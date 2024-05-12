using System;
using System.Collections;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using static UnityEditor.FilePathAttribute;
using UnityEngine.UIElements;
using Palmmedia.ReportGenerator.Core.Common;
using UnityEngine.SceneManagement;
//using System.Numerics;

public class InputFromTcp
{
    public InputFromTcp()
    {
        this.thrust = 0;
        this.rollPitch = new float[] { 0f, 0f };
        this.yaw = 0;
        this.toggleFlaps = false;
    }
    public float thrust;
    public float[] rollPitch;
    public float yaw;
    public bool toggleFlaps;
}

public class DataToSend
{
    public DataToSend(
            float playerPlanePositionx,
            float playerPlanePositiony,
            float playerPlanePositionz,
            float playerPlaneEulerRotationx,
            float playerPlaneEulerRotationy,
            float playerPlaneEulerRotationz,
            float playerPlaneForwardVelocity,
            float playerPlaneGForce,
            float enemyPlanePositionx,
            float enemyPlanePositiony,
            float enemyPlanePositionz,
            float enemyPlaneEulerRotationx,
            float enemyPlaneEulerRotationy,
            float enemyPlaneEulerRotationz,
            float enemyPlaneForwardVelocity,
            float enemyPlaneGForce,
            float relativePositionx,
            float relativePositiony,
            float relativePositionz,
            float enemyAngularVelocityx,
            float enemyAngularVelocityy,
            float enemyAngularVelocityz,
            float enemyThrustValue,
            string endGame,
            float reward
            )
    {
        this.playerPlanePositionx = playerPlanePositionx;
        this.playerPlanePositiony = playerPlanePositiony;
        this.playerPlanePositionz = playerPlanePositionz;
        this.playerPlaneEulerRotationx = playerPlaneEulerRotationx;
        this.playerPlaneEulerRotationy = playerPlaneEulerRotationy;
        this.playerPlaneEulerRotationz = playerPlaneEulerRotationz;
        this.playerPlaneForwardVelocity = playerPlaneForwardVelocity;
        this.playerPlaneGForce = playerPlaneGForce;
        this.enemyPlanePositionx = enemyPlanePositionx;
        this.enemyPlanePositiony = enemyPlanePositiony;
        this.enemyPlanePositionz = enemyPlanePositionz;
        this.enemyPlaneEulerRotationx = enemyPlaneEulerRotationx;
        this.enemyPlaneEulerRotationy = enemyPlaneEulerRotationy;
        this.enemyPlaneEulerRotationz = enemyPlaneEulerRotationz;
        this.enemyPlaneForwardVelocity = enemyPlaneForwardVelocity;
        this.enemyPlaneGForce = enemyPlaneGForce;
        this.relativePositionx = relativePositionx;
        this.relativePositiony = relativePositiony;
        this.relativePositionz = relativePositionz;
        this.enemyAngularVelocityx = enemyAngularVelocityx;
        this.enemyAngularVelocityy = enemyAngularVelocityy;
        this.enemyAngularVelocityz = enemyAngularVelocityz;
        this.enemyThrustValue = enemyThrustValue;
        this.endGame = endGame;
        this.reward = reward;
    }

    public float playerPlanePositionx;
    public float playerPlanePositiony;
    public float playerPlanePositionz;
    public float playerPlaneEulerRotationx;
    public float playerPlaneEulerRotationy;
    public float playerPlaneEulerRotationz;
    public float playerPlaneForwardVelocity;
    public float playerPlaneGForce;
    public float enemyPlanePositionx;
    public float enemyPlanePositiony;
    public float enemyPlanePositionz;
    public float enemyPlaneEulerRotationx;
    public float enemyPlaneEulerRotationy;
    public float enemyPlaneEulerRotationz;
    public float enemyPlaneForwardVelocity;
    public float enemyPlaneGForce;
    public float relativePositionx;
    public float relativePositiony;
    public float relativePositionz;
    public float enemyAngularVelocityx;
    public float enemyAngularVelocityy;
    public float enemyAngularVelocityz;
    public float enemyThrustValue;
    public string endGame;
    public float reward;
}

public class Sender : MonoBehaviour
{
    private TcpClient client;
    private string serverIp = "127.0.0.1"; // Python TCP sunucusunun IP adresi
    private int port = 8888; // TCP baðlantý noktasý
    public bool connected = false;
    public InputFromTcp input;

    public event Action<Vector2> RollPitchEvent;
    public event Action<float> ThrustEvent;
    public event Action<float> YawEvent;

    private RewardCalculator rewardCalculator;

    private GameObject playerPlane;
    private Plane playerPlaneScript;
    Vector3 playerPlanePosition;
    Vector3 playerPlaneEulerRotation;
    Vector3 playerPlaneVelocity;
    float playerPlaneGForce;

    private GameObject enemyPlane;
    private Plane enemyPlaneScript;
    Vector3 enemyPlanePosition;
    Vector3 enemyPlaneEulerRotation;
    Vector3 enemyPlaneVelocity;
    float enemyPlaneGForce;

    Vector3 relativePosition;

    Vector3 enemyAngularVelocity;

    private float enemyThrustValue;

    void Start()
    {
        rewardCalculator = gameObject.GetComponent<RewardCalculator>();
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        enemyPlane = GameObject.FindWithTag("EnemyPlane");
        playerPlaneScript = playerPlane.GetComponent<Plane>();
        enemyPlaneScript = enemyPlane.GetComponent<Plane>();

        playerPlanePosition = playerPlane.transform.position;
        playerPlaneEulerRotation = playerPlane.transform.rotation.eulerAngles;
        playerPlaneVelocity = playerPlaneScript.GetVelocity();
        playerPlaneGForce = playerPlaneScript.GetLocalGForce();

        enemyPlanePosition = enemyPlane.transform.position;
        enemyPlaneEulerRotation = enemyPlane.transform.rotation.eulerAngles;
        enemyPlaneVelocity = enemyPlaneScript.GetVelocity();
        enemyPlaneGForce = enemyPlaneScript.GetLocalGForce();

        enemyAngularVelocity = enemyPlaneScript.GetAngularVelocity();
        enemyThrustValue = enemyPlaneScript.GetThrustValue();

        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIp, port);
            connected = true;
            Debug.Log("Connected to server for sending and receiving data.");

            // Pozisyon verisi göndermek için bir döngü baþlat
            // Thread t = new Thread(DataReceiver);
            // t.Start();
            // StartCoroutine(KeyboardReceiver());
        } catch (Exception e) {
            Debug.LogError($"Failed to connect to server: {e}");
        }
    }

    void DataReceiver()
    {
        while (connected)
        {
            string keyboardData = ReceiveData();
            this.input = JsonUtility.FromJson<InputFromTcp>(keyboardData);
        }
    }

    void ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(InputFromTcp input) 
    {
        if (input.thrust != 0) {
            ThrustEvent.Invoke(input.thrust);
        } else if (!(input.rollPitch[0] == 0.0 && input.rollPitch[1] == 0.0)) {
            RollPitchEvent.Invoke(new Vector2(input.rollPitch[0], input.rollPitch[1]));
        } else if (input.yaw != 0) {
            YawEvent.Invoke(input.yaw);
        }
    }

    string ReceiveData()
    {
        try
        {
            byte[] buffer = new byte[1024];
            int bytesRead = client.GetStream().Read(buffer, 0, buffer.Length);
            string receivedData = Encoding.ASCII.GetString(buffer, 0, bytesRead);
            // Debug.Log("Received:" + receivedData);
            return receivedData;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error while receiving data: {e}");
            return null;
        }
    }

    void SendData(bool gameOver=false)
    {
        byte[] jsonData;
        DataToSend data = new DataToSend(
                Helper.MinMaxNormalize(playerPlanePosition.x, Helper.GlobalPosMinXZ, Helper.GlobalPosMaxXZ),
                Helper.MinMaxNormalize(playerPlanePosition.y, Helper.GlobalPosMinY, Helper.GlobalPosMaxY),
                Helper.MinMaxNormalize(playerPlanePosition.z, Helper.GlobalPosMinXZ, Helper.GlobalPosMaxXZ),
                // TODO
                // Euler angle çöz
                Helper.MinMaxNormalize(playerPlaneEulerRotation.x, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(playerPlaneEulerRotation.y, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(playerPlaneEulerRotation.z, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(playerPlaneVelocity.z, Helper.SpeedMin, Helper.SpeedMax),
                Helper.MinMaxNormalize(playerPlaneGForce, Helper.GForceMin, Helper.GForceMax),
                Helper.MinMaxNormalize(enemyPlanePosition.x, Helper.GlobalPosMinXZ, Helper.GlobalPosMaxXZ),
                Helper.MinMaxNormalize(enemyPlanePosition.y, Helper.GlobalPosMinXZ, Helper.GlobalPosMaxXZ),
                Helper.MinMaxNormalize(enemyPlanePosition.z, Helper.GlobalPosMinXZ, Helper.GlobalPosMaxXZ),
                Helper.MinMaxNormalize(enemyPlaneEulerRotation.x, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(enemyPlaneEulerRotation.y, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(enemyPlaneEulerRotation.z, Helper.EulerMin, Helper.EulerMax),
                Helper.MinMaxNormalize(enemyPlaneVelocity.z, Helper.SpeedMin, Helper.SpeedMax),
                Helper.MinMaxNormalize(enemyPlaneGForce, Helper.GForceMin, Helper.GForceMax),
                Helper.MinMaxNormalize(relativePosition.x, Helper.RelativePosMinXZ, Helper.RelativePosMaxXZ),
                Helper.MinMaxNormalize(relativePosition.y, Helper.RelativePosMinY, Helper.RelativePosMaxY),
                Helper.MinMaxNormalize(relativePosition.z, Helper.RelativePosMinXZ, Helper.RelativePosMaxXZ),
                Helper.MinMaxNormalize(enemyAngularVelocity.x, Helper.AngularVelMinX, Helper.AngularVelMaxX),
                Helper.MinMaxNormalize(enemyAngularVelocity.y, Helper.AngularVelMinY, Helper.AngularVelMaxY),
                Helper.MinMaxNormalize(enemyAngularVelocity.z, Helper.AngularVelMinZ, Helper.AngularVelMaxZ),
                enemyThrustValue, // Already between 0 and 1
                "CONGAME",
                Helper.MinMaxNormalize(rewardCalculator.GetReward(), 
                        -rewardCalculator.GetRewardFactorsTotal(), 
                        rewardCalculator.GetRewardFactorsTotal()));

        if (gameOver)
        {
            data.endGame = "ENDGAME";
            string json = JsonUtility.ToJson(data);
            jsonData = Encoding.ASCII.GetBytes(json);
            client.GetStream().Write(jsonData, 0, jsonData.Length);
        }
        else
        {
            string json = JsonUtility.ToJson(data);
            jsonData = Encoding.ASCII.GetBytes(json);
            client.GetStream().Write(jsonData, 0, jsonData.Length);
            // Debug.Log("JSON Sent: " + json);
        }
    }

    public void StartNewGame()
    {
        if (connected) {
            connected = false; 
            SendData(true);
            Disconnect();
        }
        int activeSceneIndex = SceneManager.GetActiveScene().buildIndex;
        SceneManager.LoadScene(activeSceneIndex);
    }

    void OnDestroy()
    {
        Disconnect();
    }

    void Disconnect()
    {
        connected = false;
        if (client != null)
        {
            client.Close();
        }
    }
   


    void FixedUpdate()
    {
        playerPlanePosition = playerPlane.transform.position;
        playerPlaneEulerRotation = playerPlane.transform.rotation.eulerAngles;
        playerPlaneVelocity = playerPlaneScript.GetVelocity();
        playerPlaneGForce = playerPlaneScript.GetLocalGForce();

        enemyPlanePosition = enemyPlane.transform.position;
        enemyPlaneEulerRotation = enemyPlane.transform.rotation.eulerAngles;
        enemyPlaneVelocity = enemyPlaneScript.GetVelocity();
        enemyPlaneGForce = enemyPlaneScript.GetLocalGForce();

        relativePosition = playerPlanePosition - enemyPlanePosition;

        enemyAngularVelocity = enemyPlaneScript.GetAngularVelocity();

        enemyThrustValue = enemyPlaneScript.GetThrustValue();

        if (connected) 
        {
            SendData();
            string keyboardData = ReceiveData();
            this.input = JsonUtility.FromJson<InputFromTcp>(keyboardData);
            ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(this.input);
        }
    }
}
