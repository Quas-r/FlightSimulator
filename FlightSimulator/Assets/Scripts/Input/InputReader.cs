using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

[CreateAssetMenu(menuName = "InputReader")]
public class InputReader : ScriptableObject, Controls.IPlaneActions
{
    Controls controls;

    private void OnEnable() {
        if (controls == null) {
            controls = new Controls();
            controls.Plane.SetCallbacks(this);
            controls.Plane.Enable();
        }
    }

    public event Action<float> ThrustEvent;

    public void OnCamera(InputAction.CallbackContext context)
    {
        Debug.Log("Test");
    }

    public void OnRollPitch(InputAction.CallbackContext context)
    {
        Debug.Log($"Action: RollPitch, Phase: {context.phase}, Input: {context.ReadValue<Vector2>()}");
    }

    public void OnThrust(InputAction.CallbackContext context)
    {
        ThrustEvent?.Invoke(context.ReadValue<float>());
    }

    public void OnToggleFlaps(InputAction.CallbackContext context)
    {
    }

    public void OnYaw(InputAction.CallbackContext context)
    {
    }
}
