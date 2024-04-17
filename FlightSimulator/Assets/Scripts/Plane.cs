using UnityEngine;
using UnityEngine.SceneManagement;

public class Plane : MonoBehaviour
{
    [SerializeField]
    InputReader inputReader;
    [SerializeField]
    float maxThrust;
    [SerializeField]
    float thrustChangeSpeed;
    [SerializeField]
    float inducedDrag;
    [SerializeField]
    AnimationCurve liftAOACurve;
    [SerializeField]
    AnimationCurve inducedDragCurve;
    [SerializeField]
    AnimationCurve rudderAOACurve;
    [SerializeField]
    AnimationCurve rudderInducedDragCurve;
    [SerializeField]
    float liftPower;
    [SerializeField]
    float rudderPower;
    [SerializeField]
    AnimationCurve dragRight;
    [SerializeField]
    AnimationCurve dragLeft;
    [SerializeField]
    AnimationCurve dragTop;
    [SerializeField]
    AnimationCurve dragBottom;
    [SerializeField]
    AnimationCurve dragForward;
    [SerializeField]
    AnimationCurve dragBackward;
    [SerializeField]
    AnimationCurve steeringCurve;
    [SerializeField]
    Vector3 turnSpeed;
    [SerializeField]
    Vector3 turnAcceleration;
    Rigidbody rb;
    Vector3 velocity;
    Vector3 localVelocity;
    Vector3 lastVelocity;
    Vector3 localAngularVelocity;
    static Vector3 localGForce;
    static float gForce;
    float angleOfAttack;
    float angleOfAttackYaw;
    Vector3 controlInput;
    float thrustInput;
    float thrustValue;
    Sender sender;
    

    // Start is called before the first frame update
    void Start()
    {
        
        rb = gameObject.GetComponent<Rigidbody>();
        sender = gameObject.GetComponent<Sender>();
        inputReader.ThrustEvent += HandleThrustInput;
        inputReader.RollPitchEvent += HandleRollPitchInput;
        inputReader.YawEvent += HandleYawInput;
        sender.ThrustEvent += HandleThrustInput;
        sender.RollPitchEvent += HandleRollPitchInput;
        sender.YawEvent += HandleYawInput;

        // Add very small torque to the wheels in order to work around a bug
        foreach (WheelCollider w in GetComponentsInChildren<WheelCollider>())
        {
            w.motorTorque = 0.00001f;
        }
    }

    private void CalculateState()
    {
        Quaternion invRotation = Quaternion.Inverse(transform.rotation);
        velocity = rb.velocity;
        localVelocity = invRotation * velocity;
        localAngularVelocity = invRotation * rb.angularVelocity;

        CalculateAngleOfAttack();
    }

    public void HandleThrustInput(float thrustInputRead)
    {
        thrustInput = thrustInputRead;
    }

    public void HandleRollPitchInput(Vector2 rollPitchInputRead)
    {
        controlInput = new Vector3(rollPitchInputRead.y, controlInput.y, -rollPitchInputRead.x);
    }

    public void HandleYawInput(float yawInputRead)
    {
        controlInput = new Vector3(controlInput.x, yawInputRead, controlInput.z);
    }

    private void UpdateThrottle(float dt)
    {
        float target = 0;
        if (thrustInput > 0) target = 1;
        thrustValue = Helper.MoveTo(thrustValue, target, thrustChangeSpeed * Mathf.Abs(thrustInput), dt);
    }

    private void ApplyThrottle()
    {
        rb.AddRelativeForce(thrustValue * maxThrust * Vector3.forward);
    }

    private void UpdateLift()
    {
        Vector3 lift = CalculateLift(
            angleOfAttack,
            Vector3.right,
            liftPower,
            liftAOACurve,
            inducedDragCurve
        );

        Vector3 yawForce = CalculateLift(
            angleOfAttackYaw,
            Vector3.up,
            rudderPower,
            rudderAOACurve,
            rudderInducedDragCurve
        );

        rb.AddRelativeForce(lift);
        rb.AddRelativeForce(yawForce);
    }

    private void CalculateAngleOfAttack()
    {
        angleOfAttack = Mathf.Atan2(-localVelocity.y, localVelocity.z);
        angleOfAttackYaw = Mathf.Atan2(localVelocity.x, localVelocity.z);
    }

    private Vector3 CalculateLift(float angleOfAttack, Vector3 crossAxis, float liftPower, AnimationCurve aoaCurve, AnimationCurve inducedDragCurve)
    {
        Vector3 liftVelocity = Vector3.ProjectOnPlane(localVelocity, crossAxis);
        float lvSquared = liftVelocity.sqrMagnitude;

        float liftCoefficient = aoaCurve.Evaluate(angleOfAttack * Mathf.Rad2Deg);
        float liftForce = liftCoefficient * lvSquared * liftPower;

        Vector3 liftDirection = Vector3.Cross(liftVelocity.normalized, crossAxis);
        Vector3 lift = liftDirection * liftForce;

        float dragForce = liftCoefficient * liftCoefficient;
        Vector3 dragDirection = -liftVelocity.normalized;
        Vector3 inducedDrag = dragDirection * dragForce * lvSquared * this.inducedDrag * inducedDragCurve.Evaluate(Mathf.Max(0, localVelocity.z));

        return lift + inducedDrag;
    }

    private void CalculateGForce(float dt)
    {
        Quaternion invRotation = Quaternion.Inverse(transform.rotation);
        Vector3 acceleration = (velocity - lastVelocity) / dt;
        localGForce = invRotation * acceleration;
        lastVelocity = velocity;
        gForce = localGForce.y /  9.81f;
        if (gForce > 9.0f && transform.position.y > 3.0f)
        {
            
            StartNewGame();

        }
    }

    public void DetectCollision()
    {
        RaycastHit hit;
        float raycastDistance = 10.0f; 

        if (Physics.Raycast(transform.position, Vector3.down, out hit, raycastDistance))
        {
            
            if (hit.collider.CompareTag("Ground")) 
            {

                StartNewGame();
              
               
            }
        }
}

    public void StartNewGame()
    {
        
        if (sender.connected)
        {
            sender.gameNotOver = false;
            while (!sender.gameNotOver); // Stupid
        }
        int activeSceneIndex = SceneManager.GetActiveScene().buildIndex;
        SceneManager.LoadScene(activeSceneIndex);
    }


    public static float GetLocalGForce()
    {
  
        return gForce;
    }


    private void UpdateDrag()
    {
        float lvSquared = localVelocity.sqrMagnitude;
        Vector3 dragCoefficient = Helper.Scale6(
            localVelocity.normalized,
            dragRight.Evaluate(Mathf.Abs(localVelocity.x)),
            dragLeft.Evaluate(Mathf.Abs(localVelocity.x)),
            dragTop.Evaluate(Mathf.Abs(localVelocity.y)),
            dragBottom.Evaluate(Mathf.Abs(localVelocity.y)),
            dragForward.Evaluate(Mathf.Abs(localVelocity.z)),
            dragBackward.Evaluate(Mathf.Abs(localVelocity.z))
        );

        Vector3 dragForce = dragCoefficient.magnitude * lvSquared * -localVelocity.normalized;

        rb.AddRelativeForce(dragForce);
    }

    private void UpdateSteering(float dt)
    {
        float speed = Mathf.Max(0, localVelocity.z);
        float steeringPower = steeringCurve.Evaluate(speed);

        Vector3 targetAV = Vector3.Scale(controlInput, steeringPower * turnSpeed);
        Vector3 angularVelocity = localAngularVelocity * Mathf.Rad2Deg;

        Vector3 correction = new Vector3(
            CalculateSteering(dt, angularVelocity.x, targetAV.x, turnAcceleration.x * steeringPower),
            CalculateSteering(dt, angularVelocity.y, targetAV.y, turnAcceleration.y * steeringPower),
            CalculateSteering(dt, angularVelocity.z, targetAV.z, turnAcceleration.z * steeringPower)
        );

        rb.AddRelativeTorque(correction * Mathf.Deg2Rad, ForceMode.VelocityChange);
    }

    private float CalculateSteering(float dt, float angularVelocity, float targetVelocity, float acceleration)
    {
        float error = targetVelocity - angularVelocity;
        float acc = acceleration * dt;
        return Mathf.Clamp(error, -acc, acc);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        
        CalculateState();
        CalculateGForce(Time.deltaTime);
        UpdateThrottle(Time.deltaTime);
        ApplyThrottle();
        UpdateLift();
        UpdateDrag();
        UpdateSteering(Time.deltaTime);
        DetectCollision();
        //Debug.Log(rb.velocity.magnitude);
    }

}
