using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Xml.Serialization;

namespace NeuralNetwork
{
    [Serializable]
    public class Layer
    {
        public Neuron[] neurons { get; set; }
        const double LEARNING_RATE = 0.05;

        private Layer()
        {

        }

        public Layer(int nCount)
        {
            neurons = new Neuron[nCount];
            for (int i = 0; i < nCount; i++)
            {
                neurons[i] = new Neuron();
            }
        }

        public void SaveLayer(string path)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate))
            {
                formatter.Serialize(fs, this);
            }
            Console.WriteLine("serialized");
        }

        public Layer LoadLayer(string path)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            Layer layer;
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                layer = formatter.Deserialize(fs) as Layer;
            }
            Console.WriteLine("deserialized");
            return layer;
        }

        public void InitNetwork()
        {
            Random rnd = new Random();
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < 28 * 28; j++)
                {
                    neurons[i].Input[j] = 0.00;
                    neurons[i].Weight[j] = rnd.NextDouble();
                }
                neurons[i].Output = 0.00;
            }
        }

        public void InitNetwork(Layer layer)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < 28 * 28; j++)
                {
                    neurons[i].Input[j] = layer.neurons[i].Input[j];
                    neurons[i].Weight[j] = layer.neurons[i].Weight[j];
                }
                neurons[i].Output = layer.neurons[i].Output;
            }
        }


        public bool TrainLayer(byte[] data, int lbl)
        {
            int[] vector = GetOutputVector(lbl);
            for (int i = 0; i < neurons.Length; i++)
            {
                TrainNeuron(neurons[i], data, vector[i]);
            }

            int predictedNum = GetLayerPrediction(this);

            if (predictedNum != lbl)
            {
                return false;
            }
            return true;
        }

        public bool TestLayer(byte[] data, int lbl)
        {
            int[] vector = GetOutputVector(lbl);
            for (int i = 0; i < neurons.Length; i++)
            {
                TestNeuron(neurons[i], data, vector[i]);
            }

            int predictedNum = GetLayerPrediction(this);
            if (predictedNum != lbl)
            {
                return false;
            }
            return true;
        }

        private void TestNeuron(Neuron neuron, byte[] data, int target)
        {
            SetNeuronInput(neuron, data);
            CalcNeuronOutput(neuron);
        }
        private void TrainNeuron(Neuron neuron, byte[] data, int target)
        {
            SetNeuronInput(neuron, data);
            CalcNeuronOutput(neuron);

            double error = GetNeuronError(neuron, target);
            UpdateNeuronWeights(neuron, error);
        }

        private int[] GetOutputVector(int lbl)
        {
            int[] vector = new int[10];
            for (int i = 0; i < 10; i++)
            {
                vector[i] = lbl == i ? 1 : 0;
            }
            return vector;
        }

        private int GetLayerPrediction(Layer layer)
        {
            double maxOut = 0;
            int maxInd = 0;

            for (int i = 0; i < layer.neurons.Length; i++)
            {
                if (layer.neurons[i].Output > maxOut)
                {
                    maxOut = layer.neurons[i].Output;
                    maxInd = i;
                }
            }
            return maxInd;
        }

        private void UpdateNeuronWeights(Neuron neuron, double error)
        {
            for (int i = 0; i < neuron.Input.Length; i++)
            {
                neuron.Weight[i] += LEARNING_RATE * neuron.Input[i] * error;
            }
        }

        private void CalcNeuronOutput(Neuron neuron)
        {
            neuron.Output = 0;
            for (int i = 0; i < neuron.Input.Length; i++)
            {
                neuron.Output += neuron.Input[i] * neuron.Weight[i];
            }
            neuron.Output = 1.00 / (1.00 + Math.Exp(-neuron.Output));
            if (neuron.Output < 0) 
            Console.WriteLine(neuron.Output);
        }

        private double GetNeuronError(Neuron neuron, int target)
        {
            double error = target - neuron.Output;
            return error;
        }

        private void SetNeuronInput(Neuron neuron, byte[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                neuron.Input[i] = data[i] > 0 ? 1 : 0;
            }
        }
        public void Write(byte[] pixels)
        {
            string s = "";
            for (int i = 1; i < (28 * 28) + 1; i++)
            {
                if (pixels[i - 1] == 0) //белый 
                    s += "@";
                else if (pixels[i - 1] == 255) //черный 
                    s += "O";
                else
                    s += "."; //оттенки черного 
                if (i % 28 == 0)
                    s += "\n";
            }
            Console.WriteLine(s + "\n");
        }
    }
}