using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RewardCalculator : MonoBehaviour
{
  
    private const float highAltitudeSpeed = 925f; // km/s
    private const float lowAltitudeSpeed = 800f; // km/s
    private const float mediumAltitudeSpeed = 900f; // km/s
    private const float highAltitudeLimit = 12200f; // m

    private const float maxGForce = 7f;

    private const float aim120SidewinderLowDistance = 1000f; // km
    private const float aim120SidewinderHighDistance = 34500f; // km


    public static float CalculateReward(Transform civilianTransform, Transform enemyTransform, float velocity, float gForce)
    {
        Vector3 los = civilianTransform.position - enemyTransform.position;
        float distance = los.magnitude;

        float cosAngleAA = Vector3.Dot(civilianTransform.forward, los) / (civilianTransform.forward.magnitude * los.magnitude);
        float aaAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleAA, -1f, 1f));

        float cosAngleATA = Vector3.Dot(enemyTransform.forward, -los) / (enemyTransform.forward.magnitude * los.magnitude);
        float ataAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleATA, -1f, 1f));

        float reward = 0f;

        if (distance < 0.1f)
        {
            reward -= 10f;
        }
        else if (distance >= aim120SidewinderLowDistance && distance <= aim120SidewinderHighDistance)
        {
            reward += 10f;
            if (Mathf.Abs(aaAngle) < 1f && Mathf.Abs(ataAngle) < 1f) // Adjust angle thresholds if needed
            {
                reward += 10f;
            }
        }
        else if (distance > aim120SidewinderHighDistance)
        {
            if (Mathf.Abs(aaAngle) < 60f && Mathf.Abs(ataAngle) < 30f)
            {
                reward += 2f;
            }
            else if (Mathf.Abs(ataAngle) > 120f && Mathf.Abs(aaAngle) > 150f)
            {
                reward -= 2f;
            }
        }

        if (enemyTransform.position.y < highAltitudeLimit &&
            velocity > lowAltitudeSpeed && velocity < mediumAltitudeSpeed)
        {
            reward += 1f;
        }
        else if (enemyTransform.position.y >= highAltitudeLimit && Mathf.Abs(velocity - highAltitudeSpeed) < 0.01f)
        {
            reward += 1f;
        }
        else
        {
            reward -= 1f;
        }

        if (gForce <= maxGForce)
        {
            reward += 5f;
        }

        return reward;
    }

}
