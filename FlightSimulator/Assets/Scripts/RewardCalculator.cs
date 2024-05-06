using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class RewardCalculator
{
    private const float highAltitudeSpeed = 925f; // km/s
    private const float lowAltitudeSpeed = 800f; // km/s
    private const float mediumAltitudeSpeed = 900f; // km/s
    private const float highAltitudeLimit = 12200f; // m

    private const float maxGForce = 7f;

    private const float minDistance = 5f;

    private const float aim120SidewinderShortDistance = 100f; // km
    private const float aim120SidewinderLongDistance = 300f; // km






    public static float CalculateReward(Transform playerPlaneTransform, Transform enemyPlaneTransform, float enemyPlaneForwardVelocity, float enemyPlaneGForce)
    {
        Vector3 los = playerPlaneTransform.position - enemyPlaneTransform.position;
        float distance = los.magnitude;
        Debug.Log(los);

        Debug.Log(playerPlaneTransform.forward);
        float cosAngleAA = Vector3.Dot(playerPlaneTransform.forward, los) / (playerPlaneTransform.forward.magnitude * los.magnitude);
        float aaAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleAA, -1f, 1f));
        Debug.Log(aaAngle);

        float cosAngleATA = Vector3.Dot(enemyPlaneTransform.forward, los) / (enemyPlaneTransform.forward.magnitude * los.magnitude);
        float ataAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleATA, -1f, 1f));
        Debug.Log(ataAngle);

        float reward = 0f;

        if (distance < minDistance)
        {
            reward -= 10f;
        }
        else
        {
            float normalizedDistanceReward = Mathf.Exp(-distance / 100f);
            reward += normalizedDistanceReward * 10f;
        }

        if (ataAngle > 90)
        {
            Debug.Log(ataAngle);
            reward += -10f * (ataAngle - 90f) / 90f;
        }
        else
        {
            float angleReward = Mathf.Exp(-Mathf.Pow(aaAngle / 45f, 2)) + Mathf.Exp(-Mathf.Pow(ataAngle / 45f, 2));
            reward += angleReward * 5f;



            // Hýz için sürekli ödül hesaplama
            if (enemyPlaneTransform.position.y < highAltitudeLimit)
            {
                float speedDifference = Mathf.Abs(enemyPlaneForwardVelocity - mediumAltitudeSpeed);
                reward += Mathf.Exp(-Mathf.Pow(speedDifference / 50f, 2)) * 2f;
            }
            else
            {
                float speedDifference = Mathf.Abs(enemyPlaneForwardVelocity - highAltitudeSpeed);
                reward += Mathf.Exp(-Mathf.Pow(speedDifference / 50f, 2)) * 2f;
            }


            float gForceDifference = Mathf.Abs(enemyPlaneGForce - maxGForce);
            float gForceReward = Mathf.Exp(-gForceDifference / (maxGForce * 0.1f));
            reward += gForceReward * 5f;

            
        }
        return reward;
    }
}
