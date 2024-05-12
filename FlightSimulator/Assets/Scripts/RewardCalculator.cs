using System;
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

    private const float minDistance = 5f;

    private const float aim120SidewinderShortDistance = 100f; // km
    private const float aim120SidewinderLongDistance = 300f; // km

    private float reward;
    private GameObject playerPlane;
    private Plane playerPlaneScript;
    private GameObject enemyPlane;
    private Plane enemyPlaneScript;

    [SerializeField]
    private AnimationCurve aaAngleRewardCurve;
    [SerializeField]
    private float aaAngleRewardFactor;

    [SerializeField]
    private AnimationCurve ataAngleRewardCurve;
    [SerializeField]
    private float ataAngleRewardFactor;

    [SerializeField]
    private AnimationCurve distanceRewardCurve;
    [SerializeField]
    private float distanceRewardFactor;

    [SerializeField]
    private AnimationCurve speedDiffRewardCurve;
    [SerializeField]
    private float speedDiffRewardFactor;

    [SerializeField]
    private AnimationCurve gForceRewardCurve;
    [SerializeField]
    private float gForceRewardFactor;

    [SerializeField]
    private AnimationCurve optimumSpeedForAltitudeCurve;
    [SerializeField]
    private AnimationCurve altitudeSpeedDiscrepancyRewardCurve;
    [SerializeField]
    private float altitudeSpeedDiscrepancyRewardFactor;

    private float rewardFactorsTotal;

    void Start()
    {
        rewardFactorsTotal = aaAngleRewardFactor + 
            ataAngleRewardFactor + 
            distanceRewardFactor + 
            speedDiffRewardFactor + 
            gForceRewardFactor + 
            altitudeSpeedDiscrepancyRewardFactor;
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        enemyPlane = GameObject.FindWithTag("EnemyPlane");
        playerPlaneScript = playerPlane.GetComponent<Plane>();
        enemyPlaneScript = enemyPlane.GetComponent<Plane>();
    }

    public float CalculateReward()
    {
        Vector3 los = playerPlane.transform.position - enemyPlane.transform.position;
        float distance = los.magnitude;

        float cosAngleAA = Vector3.Dot(playerPlane.transform.forward, los) / (playerPlane.transform.forward.magnitude * los.magnitude);
        float aaAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleAA, -1f, 1f));

        float cosAngleATA = Vector3.Dot(enemyPlane.transform.forward, los) / (enemyPlane.transform.forward.magnitude * los.magnitude);
        float ataAngle = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp(cosAngleATA, -1f, 1f));
        string log = "AA: " + aaAngle + ", ATA: " + ataAngle + ", Dist: " + distance + ", Speed: " + enemyPlaneScript.localVelocity.z + " | Rewards -> ";

        float reward = 0f;
        float x;

        x = distanceRewardCurve.Evaluate(distance) * distanceRewardFactor;
        log += "Dist: " + x + ", ";
        reward += x;
        x = aaAngleRewardCurve.Evaluate(aaAngle) * aaAngleRewardFactor;
        log += "AA: " + x + ", ";
        reward += x;
        x = ataAngleRewardCurve.Evaluate(ataAngle) * ataAngleRewardFactor;
        log += "ATA: " + x + ", ";
        reward += x;
        x = speedDiffRewardCurve.Evaluate(enemyPlaneScript.localVelocity.z - playerPlaneScript.localVelocity.z) * speedDiffRewardFactor;
        log += "Speed diff: " + x + ", ";
        reward += x;
        float altitudeSpeedDiscrepancy = optimumSpeedForAltitudeCurve.Evaluate(enemyPlane.transform.position.y) - enemyPlaneScript.localVelocity.z;
        x = altitudeSpeedDiscrepancyRewardCurve.Evaluate(altitudeSpeedDiscrepancy) * altitudeSpeedDiscrepancyRewardFactor;
        log += "Alt-speed disc: " + x + ", ";
        reward += x;
        x = gForceRewardCurve.Evaluate(enemyPlaneScript.GetLocalGForce()) * gForceRewardFactor;
        reward += x;
        log += "G-force: " + x + ", TOTAL: " + reward + ", NORMALIZED: " + Helper.MinMaxNormalize(reward, -rewardFactorsTotal, rewardFactorsTotal);

        Debug.Log(log);

        return reward;
    }

    public float GetReward()
    {
        return this.reward;
    }

    public float GetRewardFactorsTotal()
    {
        return this.rewardFactorsTotal;
    }

    void FixedUpdate()
    {
        this.reward = CalculateReward();
    }
}
