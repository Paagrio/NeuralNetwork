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
    const double LEARNING_RATE = 0.05;//скорость обучения

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

    public void InitNetwork() //инициализация из случайных весов
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

    public void InitNetwork(Layer layer) //инициализация из параметра
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

    //обучение
    public bool TrainLayer(byte[] data, int lbl)//data-массив пикселей, lbl-истинное число
    {
      int[] vector = GetOutputVector(lbl);
      Console.WriteLine("input: " + lbl);

      for (int i = 0; i < neurons.Length; i++)
      {
        TrainNeuron(neurons[i], data, vector[i]);
      }

      int predictedNum = GetLayerPrediction(this);
      Console.WriteLine("Prediction: " + predictedNum);
      if (predictedNum != lbl)
      {
        return false;
      }
      return true;
    }
    //тестирование
    public bool TestLayer(byte[] data, int lbl)
    {
      int[] vector = GetOutputVector(lbl);
      Console.WriteLine("input: " + lbl);
      for (int i = 0; i < neurons.Length; i++)
      {
        TestNeuron(neurons[i], data, vector[i]);
      }

      int predictedNum = GetLayerPrediction(this);
      Console.WriteLine("Prediction: " + predictedNum);
      if (predictedNum != lbl)
      {
        return false;
      }
      return true;
    }

    private void TestNeuron(Neuron neuron, byte[] data, int target)
    {
      //в тестировании все то же самое, что и в обучении,
      //только без обновления весов
      SetNeuronInput(neuron, data);
      CalcNeuronOutput(neuron);
    }
    private void TrainNeuron(Neuron neuron, byte[] data, int target)
    {
      //тренировка отдельного нейрона
      SetNeuronInput(neuron, data);
      //расчет выхода нейрона
      CalcNeuronOutput(neuron);

      //расчет ошибки
      double error = GetNeuronError(neuron, target);
      //обновление весов
      UpdateNeuronWeights(neuron, error);
    }

    private int[] GetOutputVector(int lbl)
    {
      //получаем вектор по цифре
      //например если число 6, вектор будет такой
      //[0,0,0,0,0,0,1,0,0,0]
      int[] vector = new int[10];
      for (int i = 0; i < 10; i++)
      {
        vector[i] = lbl == i ? 1 : 0;
      }
      return vector;
    }

    private int GetLayerPrediction(Layer layer)
    {
      //здесь мы должны выбрать индекс выходного сигнала нейрона максимально приближенного к единице.
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
      //метод обратного распространения ошибки ??????????????????????????????????????????????
      //если ошибка положительная - вес увеличится
      //при отрицательной - соответственно уменьшится
      for (int i = 0; i < neuron.Input.Length; i++)
      {
        neuron.Weight[i] += LEARNING_RATE * neuron.Input[i] * error;
      }
    }

    private void CalcNeuronOutput(Neuron neuron)
    {
      //здесь просто перемножаем вес нейрона на входной параметр и суммируем
      //для изображения 28х28 получается 28*28 итераций
      neuron.Output = 0;
      for (int i = 0; i < neuron.Input.Length; i++)
      {
        neuron.Output += neuron.Input[i] * neuron.Weight[i];
      }
      neuron.Output = 1.00 / (1.00 + Math.Exp(-neuron.Output));//сигмоид для масштабирования выходного значения
    }

    private double GetNeuronError(Neuron neuron, int target)
    {
      //ошибка будет положительной только у верного нейрона так как target будет равен 1
      //для остальных 9 нейронов ошибка будет отрицательным значением
      double error = target - neuron.Output;
      return error;
    }

    private void SetNeuronInput(Neuron neuron, byte[] data)
    {
      //изначально наше изображение состоит из значений от 0(белый) до 255(черный или оттенки серого)
      //то есть на вход нейрона мы подаем массив значений каждого отдельного пикселя
      //если пиксель равен 0, то присваиваем 0
      //все что больше - присваиваем 1
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