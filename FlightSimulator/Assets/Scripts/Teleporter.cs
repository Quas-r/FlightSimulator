using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Teleporter : MonoBehaviour
{
    private float xBound = 2000f;
    private float zBound = 2000f;
    private GameObject playerPlane;
    private GameObject enemyPlane;
    

    void Start()
    {
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        enemyPlane = GameObject.FindWithTag("EnemyPlane");
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (playerPlane.transform.position.x > xBound || playerPlane.transform.position.x < -xBound ||
                playerPlane.transform.position.z > zBound || playerPlane.transform.position.z < -zBound)
        {
            Debug.Log("Test");
            Vector3 diff = enemyPlane.transform.position - playerPlane.transform.position;
            playerPlane.transform.position = new Vector3(0, playerPlane.transform.position.y, 0);
            enemyPlane.transform.position = new Vector3(diff.x, enemyPlane.transform.position.y, diff.z);
        }
    }
}
