using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;   

public class GForceEffect : MonoBehaviour
{
    public Image gForceEffect;
    private float gForceValue;

    private GameObject playerPlane;
    private Plane planeScript;
    public float gForceThreshold = 5.0f; 
    public float changeSpeed = 2f; 
    public float returnToNormalSpeed = 1f; 
    
    private Color originalColor;
    

    public void Start()
    {
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        planeScript = playerPlane.GetComponent<Plane>();
        originalColor = gForceEffect.color;
    }
    public void FixedUpdate()
    {
        gForceValue = planeScript.GetLocalGForce();
        
        if (gForceValue >= gForceThreshold)
        {
            float darkeningRate = gForceValue / gForceThreshold;
            Color newColor = Color.Lerp(Color.black, originalColor, darkeningRate * changeSpeed);
            gForceEffect.color = newColor;
        }
      
        else
        {
            float returnToNormalRate = (gForceValue - gForceThreshold) / gForceThreshold;
            Color newColor = Color.Lerp(originalColor, Color.black, returnToNormalRate * returnToNormalSpeed);
            gForceEffect.color = newColor;
        }

       
    }
}
