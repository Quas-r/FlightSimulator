using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class CameraController : MonoBehaviour
{
    private Controls controls;

    public float mouseSensivity = 50f;

    //keeps mouse movement
    private Vector2 mouseLook;

    private float xRotation = 0f; //movement in x axis
    private float yRotation = 0f; //movement in y axis



    private void Awake()
    {
        
        controls = new Controls();
        Cursor.lockState = CursorLockMode.Locked; //locks mouse cursor
    }
    private void Update()
    {
        Look();
    }
    private void Look()
    {
        mouseLook = controls.Plane.Camera.ReadValue<Vector2>();
        float mouseX = mouseLook.x*mouseSensivity*Time.deltaTime;
        float mouseY = mouseLook.y * mouseSensivity * Time.deltaTime;

        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);

        yRotation -= mouseX;
        yRotation = Mathf.Clamp(yRotation, -90f, 90f);

        transform.localRotation = Quaternion.Euler(xRotation,-yRotation,0);
        

    }
    private void OnEnable()
    {
        controls.Enable();
    }
    private void OnDisable()
    {
        controls.Disable();
    }
}
