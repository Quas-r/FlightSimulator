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

    private const float aim120SidewinderShortDistance = 1000f; // km
    private const float aim120SidewinderLongDistance = 34500f; // km


    public static float CalculateReward(Transform playerPlaneTransform, Transform enemyPlaneTransform, float enemyPlaneForwardVelocity, float enemyPlaneGForce)
    {
        Vector3 los = playerPlaneTransform.position - enemyPlaneTransform.position;
        float distance = los.magnitude;

        float cosAngleAA = Vector3.Dot(playerPlaneTransform.forward, los) / (playerPlaneTransform.forward.magnitude * los.magnitude);
        float aaAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleAA, -1f, 1f));

        float cosAngleATA = Vector3.Dot(enemyPlaneTransform.forward, -los) / (enemyPlaneTransform.forward.magnitude * los.magnitude);
        float ataAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleATA, -1f, 1f));

        float reward = 0f;

        if (distance < 0.1f) { reward -= 10f; }
        else if (distance >= aim120SidewinderShortDistance && distance <= aim120SidewinderLongDistance)
        {
            reward += 10f;
            if (Mathf.Abs(aaAngle) < 1f && Mathf.Abs(ataAngle) < 1f) { reward += 10f; }
        }
        else if (distance > aim120SidewinderLongDistance)
        {
            if (Mathf.Abs(aaAngle) < 60f && Mathf.Abs(ataAngle) < 30f) { reward += 2f; }
            else if (Mathf.Abs(ataAngle) > 120f && Mathf.Abs(aaAngle) > 150f) { reward -= 2f; }
        }

        if (enemyPlaneTransform.position.y < highAltitudeLimit &&
            enemyPlaneForwardVelocity > lowAltitudeSpeed && enemyPlaneForwardVelocity < mediumAltitudeSpeed)
        {
            reward += 1f;
        }
        else if (enemyPlaneTransform.position.y >= highAltitudeLimit && Mathf.Abs(enemyPlaneForwardVelocity - highAltitudeSpeed) < 0.01f)
        {
            reward += 1f;
        }
        else { reward -= 1f; }

        if (enemyPlaneGForce <= maxGForce) { reward += 5f; }

        return reward;
    }

}
