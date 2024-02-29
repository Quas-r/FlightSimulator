using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;   

public class GForceEffect : MonoBehaviour
{
    public Image gForceEffect;
    private float gForceValue;

    public float gForceThreshold = 5.0f; 
    public float changeSpeed = 2f; 
    public float returnToNormalSpeed = 1f; 
    
    private Color originalColor;
    

    public void Start()
    {
        originalColor = gForceEffect.color;
    }
    public void FixedUpdate()
    {
        gForceValue = Plane.GetLocalGForce();
        
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
