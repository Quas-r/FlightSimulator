using UnityEngine;

public static class Helper
{
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
