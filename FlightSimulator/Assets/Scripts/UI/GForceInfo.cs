using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GForceInfo : MonoBehaviour
{
    //public Text enemyPlaneGForceText;
    public Text playerPlaneGForceText;

    private GameObject playerPlane;
    private Plane playerPlaneScript;

    private GameObject enemyPlane;
    private Plane enemyPlaneScript;

    private float playerPlaneGForceValue;
    private float enemyPlaneGForceValue;

    private float gForceThreshold = 5.0f;

    private float blinkDuration = 2.0f;
    private Color blinkColor = Color.red;

    private bool isBlinkingPlayer = false;

    private float elapsedTime = 0f;

    void Start()
    {
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        playerPlaneScript = playerPlane.GetComponent<Plane>();

        //enemyPlane = GameObject.FindWithTag("EnemyPlane");
        //enemyPlaneScript = enemyPlane.GetComponent<Plane>();

    } 

    public void FixedUpdate()
    {
        playerPlaneGForceValue = playerPlaneScript.GetLocalGForce();
        //enemyPlaneGForceValue = enemyPlaneScript.GetLocalGForce();

        if (playerPlaneGForceText != null)
        {
            playerPlaneGForceText.text = string.Format("Player Plane G Force: {0:0.0} G", playerPlaneGForceValue);
            
        }

        //if (enemyPlaneGForceText != null)
        //{
        //    enemyPlaneGForceText.text = string.Format("Enemy Plane G Force: {0:0.0} G", enemyPlaneGForceValue);

        //}

        if (!isBlinkingPlayer && playerPlaneGForceValue > gForceThreshold)
        {
            StartBlinking();
        }
        else if (isBlinkingPlayer && playerPlaneGForceValue <= gForceThreshold)
        {
            StopBlinking();
        }


        if (isBlinkingPlayer)
        {
            elapsedTime += Time.fixedDeltaTime;

            if (elapsedTime < blinkDuration)
            {
                float lerpValue = Mathf.PingPong(elapsedTime, blinkDuration) / blinkDuration;
                playerPlaneGForceText.color = Color.Lerp(Color.white, blinkColor, lerpValue);

            }
            else
            {
                StopBlinking();
            }
        }


        
    }
    private void StartBlinking()
    {
        
       
            isBlinkingPlayer = true;
            elapsedTime = 0f;
        
       
    }

    private void StopBlinking()
    {
        
            isBlinkingPlayer = false;
            playerPlaneGForceText.color = Color.white;
        
       
    }

    
  
}
