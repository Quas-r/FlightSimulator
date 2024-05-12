using UnityEngine;

public static class Helper
{
    
    public const float RelativePosMinXZ = -3000;
    public const float RelativePosMaxXZ = 3000;
    public const float RelativePosMinY = 0;
    public const float RelativePosMaxY = 5000;
    public const float GForceMin = -5;
    public const float GForceMax = 9;
    public const float EulerMin = 0;
    public const float EulerMax = 360;
    public const float SpeedMin = 0;
    public const float SpeedMax = 300;
    public const float AngularVelMinX = - Mathf.PI / 6;
    public const float AngularVelMinY = - Mathf.PI / 12;
    public const float AngularVelMinZ = - 3 * Mathf.PI / 2;
    public const float AngularVelMaxX = Mathf.PI / 6;
    public const float AngularVelMaxY = Mathf.PI / 12;
    public const float AngularVelMaxZ = 3 * Mathf.PI / 2;
    public const float LosAnglesMin = 0;
    public const float LosAnglesMax = 180;

    public static float MinMaxNormalize(float value, float min, float max)
    {
        return (value - min) / (max - min);
    }

    public static float MoveTo(float value, float target, float speed, float deltaTime, float min = 0, float max = 1)
    {
        float diff = target - value;
        float delta = Mathf.Clamp(diff, -speed * deltaTime, speed * deltaTime);
        return Mathf.Clamp(value + delta, min, max);
    }

    public static Vector3 Scale6(
        Vector3 vector,
        float posx, float negx,
        float posy, float negy,
        float posz, float negz
    )
    {
        Vector3 result = vector;

        if (vector.x < 0) { result.x *= negx; }
        else { result.x *= posx; }

        if (vector.y < 0) { result.y *= negy; }
        else { result.y *= posy; }

        if (vector.z < 0) { result.z *= negz; }
        else { result.z *= posz; }

        return result;
    }
}
