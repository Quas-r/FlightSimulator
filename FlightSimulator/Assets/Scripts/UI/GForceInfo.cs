using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GForceInfo : MonoBehaviour
{
    public Text gForceText;
    private float gForceValue;
    private float gForceThreshold = 5.0f;
    private float blinkDuration = 2.0f;
    private Color blinkColor = Color.red;
    private bool isBlinking = false;
    private float elapsedTime = 0f;

    public void FixedUpdate()
    {
        gForceValue = Plane.GetLocalGForce();
        

        if (gForceText != null)
        {
            gForceText.text = string.Format("G Force: {0:0.0} G", gForceValue);
        }


        if (!isBlinking && gForceValue > gForceThreshold)
        {
            StartBlinking();


        }
        else if (isBlinking && gForceValue <= gForceThreshold)
        {
            StopBlinking();
        }


        if (isBlinking)
        {
            elapsedTime += Time.fixedDeltaTime;

            if (elapsedTime < blinkDuration)
            {
                float lerpValue = Mathf.PingPong(elapsedTime, blinkDuration) / blinkDuration;
                gForceText.color = Color.Lerp(Color.white, blinkColor, lerpValue);

            }
            else
            {
                StopBlinking();
            }
        }
    }
    private void StartBlinking()
    {
        isBlinking = true;
        elapsedTime = 0f;
    }

    private void StopBlinking()
    {
        isBlinking = false;
        gForceText.color = Color.white;
    }
}
