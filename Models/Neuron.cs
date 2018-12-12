using System;

namespace NeuralNetwork
{
    [Serializable]
    public class Neuron
    {
        public double[] Input { get; set; }

        public double[] Weight { get; set; }

        public double Output { get; set; }

        public Neuron()
        {
            Input = new double[28 * 28];
            Weight = new double[28 * 28];
            Output = 0.00;
        }
    }
}