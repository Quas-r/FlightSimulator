using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

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
    Rigidbody rb;
    Vector3 velocity;
    Vector3 localVelocity;
    Vector3 lastVelocity;
    Vector3 localAngularVelocity;
    Vector3 localGForce;
    float angleOfAttack;
    float angleOfAttackYaw;
    float thrustInput;
    float thrustValue;

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        inputReader.ThrustEvent += HandleThrustInput;

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

    private void HandleThrustInput(float thrustInputRead)
    {
        thrustInput = thrustInputRead;
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
        angleOfAttack = Mathf.Atan2(localVelocity.y, localVelocity.z);
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
    }

    private void UpdateDrag()
    {
        float lvSquared = localVelocity.sqrMagnitude;
        Vector3 dragCoefficient = Helper.Scale6(
            localVelocity,
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

    // Update is called once per frame
    void FixedUpdate()
    {
        CalculateState();
        CalculateGForce(Time.deltaTime);
        UpdateThrottle(Time.deltaTime);
        ApplyThrottle();
        UpdateLift();
        UpdateDrag();
        //Debug.Log(rb.velocity.magnitude);
    }

}
