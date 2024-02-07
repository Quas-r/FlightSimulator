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

    public event Action<Vector2> CameraEvent;
    public event Action<Vector2> RollPitchEvent;
    public event Action<float> ThrustEvent;
    public event Action FlapsEvent;
    public event Action<float> YawEvent;

    public void OnCamera(InputAction.CallbackContext context)
    {
        CameraEvent.Invoke(context.ReadValue<Vector2>());
    }

    public void OnRollPitch(InputAction.CallbackContext context)
    {
        RollPitchEvent.Invoke(context.ReadValue<Vector2>());
    }

    public void OnThrust(InputAction.CallbackContext context)
    {
        ThrustEvent.Invoke(context.ReadValue<float>());
    }

    public void OnToggleFlaps(InputAction.CallbackContext context)
    {
        FlapsEvent.Invoke();
    }

    public void OnYaw(InputAction.CallbackContext context)
    {
        YawEvent.Invoke(context.ReadValue<float>());
    }
}
