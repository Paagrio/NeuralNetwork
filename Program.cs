using System;
using System.IO;
using NeuralNetwork;
using System.Drawing;
class Program
{
    static void Main(string[] args)
    {
        try
        {
            Layer nn = new Layer(10);
            nn.InitNetwork();

            #region Learning

            FileStream ifsLabels = new FileStream("train-labels.idx1-ubyte", FileMode.Open);//поток для чтения лейблов 
            FileStream ifsImages = new FileStream("train-images.idx3-ubyte", FileMode.Open); // test images 

            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // магическое число 
            int numImages = brImages.ReadInt32(); //количество изображений 
            int numRows = brImages.ReadInt32(); //количество строк в изображении 
            int numCols = brImages.ReadInt32(); //количество столбцов изображения 
            int magic2 = brLabels.ReadInt32(); //магическое число 
            int numLabels = brLabels.ReadInt32(); //количество лейблов 

            byte[] pixels = new byte[28 * 28]; //инициализация массива для хранения изображения 28x28 
            int success = 0;

            for (int di = 0; di < 60000; ++di)
            {
                byte lbl = brLabels.ReadByte(); //текущее значение лейбла 
                for (int i = 0; i < 28 * 28; ++i)
                {
                    byte b = brImages.ReadByte();
                    pixels[i] = b; //считываем байт изображения в массив 
                }
                if (nn.TrainLayer(pixels, lbl))
                {
                    success++;
                }
            }
            nn.SaveLayer("layer.dat");
            Console.WriteLine("training success rate: " + success / 60000.00 * 100.00);
            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            #endregion

            #region Testing

            ifsLabels = new FileStream("t10k-labels.idx1-ubyte", FileMode.Open);//поток для чтения лейблов 
            ifsImages = new FileStream("t10k-images.idx3-ubyte", FileMode.Open); // test images 
            brLabels = new BinaryReader(ifsLabels);
            brImages = new BinaryReader(ifsImages);

            magic1 = brImages.ReadInt32(); // магическое число 
            numImages = brImages.ReadInt32(); //количество изображений 
            numRows = brImages.ReadInt32(); //количество строк в изображении 
            numCols = brImages.ReadInt32(); //количество столбцов изображения 

            magic2 = brLabels.ReadInt32(); //магическое число 
            numLabels = brLabels.ReadInt32(); //количество лейблов

            success = 0;
            nn = nn.LoadLayer("layer.dat");
            for (int di = 0; di < 10000; ++di)
            {
                byte lbl = brLabels.ReadByte(); //текущее значение лейбла 
                for (int i = 0; i < 28 * 28; ++i)
                {
                    byte b = brImages.ReadByte();
                    pixels[i] = b; //считываем байт изображения в массив 
                }
                if (nn.TestLayer(pixels, lbl))
                {
                    success++;
                }
            }
            Console.WriteLine("Testing success rate: " + success / 10000.00 * 100.00);
            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            #endregion
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
        }
    }
}
