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

    Rigidbody rb;
    Vector3 velocity;

    Vector3 localVelocity;
    Vector3 lastVelocity;
    Vector3 localAngularVelocity;
    Vector3 localGForce;

    float thrustInput;

    float thrustValue;

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        inputReader.ThrustEvent += HandleThrustInput;
    }

    private void CalculateState() {
        Quaternion invRotation = Quaternion.Inverse(transform.rotation);
        velocity = rb.velocity;
        localVelocity = invRotation * velocity;
        localAngularVelocity = invRotation * rb.angularVelocity;

        //CalculateAngleOfAttack();
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
        Debug.Log("Thrust input: " + thrustInput + ", target value: " + target + ", thrust value: " + thrustValue);
    }

    private void ApplyThrottle() {
        rb.AddRelativeForce(thrustValue * maxThrust * Vector3.forward);
    }

    private void CalculateAngleOfAttack()
    {
        throw new NotImplementedException();
    }

    private void CalculateGForce(float dt) {
        Quaternion invRotation = Quaternion.Inverse(transform.rotation);
        Vector3 acceleration = (velocity - lastVelocity) / dt;
        localGForce = invRotation * acceleration;
        lastVelocity = velocity;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        CalculateState();
        CalculateGForce(Time.deltaTime);
        UpdateThrottle(Time.deltaTime);
        ApplyThrottle();
    }
}
