using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

var folderPath = "C:\\Users\\075693\\telent Technology Services Limited\\Software - Sample of images with issues";

try
{
    var imageFiles = Directory.GetFiles(folderPath, "*.jpg");

    foreach (var imagePath in imageFiles)
    {
        try
        {
            var numOfFaults = 0;

            // Block colours
            if (IsMostlyOneColour(imagePath).Any(x => x == true))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is mostly one colour.");
                numOfFaults++;
            }

            // Noisiness
            if (IsHeavilyNoisy(imagePath).Any(x => x == true))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is heavily noisy.");
                numOfFaults++;
            }

            // Colour tinting
            if (IsTinted(imagePath).Any(r => r == true))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is tinted.");
                numOfFaults++;
            }

            // Laplacian variance - blurriness
            if (GetLaplacianVariance(imagePath).Any(x => x < 100))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} blurry: Variance -> {variance}");
                numOfFaults++;
            }

            // Image exposure
            if (AnalyzeExposure(imagePath).Any(x => x != "Properly Exposed"))
            {
                // Console.WriteLine($"Image {Path.GetFileName(imagePath)} is over or under exposed.");
                numOfFaults++;
            }

            Console.WriteLine($"Image {Path.GetFileName(imagePath)} has {numOfFaults}/5 faults.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred while loading {Path.GetFileName(imagePath)}: {ex.Message} {ex.StackTrace}");
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"An error occurred: {ex.Message}");
}

/// <summary>
/// Blur detection
/// </summary>
static IEnumerable<double> GetLaplacianVariance(string imagePath)
{
    try
    {
        // Load the image and split
        var imageQuadrants = SplitImageIntoQuadrants(imagePath, ImreadModes.Grayscale);
        var results = new List<double>();

        foreach (var image in imageQuadrants)
        {
            using var laplacian = new Mat();

            // Calculate the Laplacian
            CvInvoke.Laplacian(image, laplacian, DepthType.Cv64F);

            // Initialize mean and stddev
            var mean = new MCvScalar();
            var stddev = new MCvScalar();

            // Calculate the variance of the Laplacian
            CvInvoke.MeanStdDev(laplacian, ref mean, ref stddev);
            results.Add(stddev.V0 * stddev.V0);
        }

        return results;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return null;
    }
}

/// <summary>
/// Determines if an image is mostly one block colour
/// </summary>
static IEnumerable<bool> IsMostlyOneColour(string imagePath)
{
    try
    {
        // Load the image an split
        var imageQuadrants = SplitImageIntoQuadrants(imagePath, ImreadModes.Color);
        var results = new List<bool>();

        foreach (var image in imageQuadrants)
        {
            // Initialize mean and stddev
            var mean = new MCvScalar();
            var stddev = new MCvScalar();

            // Calculate the mean and standard deviation of the image
            CvInvoke.MeanStdDev(image, ref mean, ref stddev);

            // If the standard deviation is very low, the image is mostly one colour
            var threshold = 20.0; // You can adjust this threshold based on your needs
            var isMostlyOneColour = stddev.V0 < threshold && stddev.V1 < threshold && stddev.V2 < threshold;

            results.Add(isMostlyOneColour);
        }

        return results;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return null;
    }
}

/// <summary>
/// Determines if an image is very heavily noisy
/// </summary>
static IEnumerable<bool> IsHeavilyNoisy(string imagePath)
{
    try
    {
        // Load the image and split
        var imageQuadrants = SplitImageIntoQuadrants(imagePath, ImreadModes.Grayscale);

        var results = new List<bool>();

        foreach (var image in imageQuadrants)
        {
            // Initialize mean and stddev
            var mean = new MCvScalar();
            var stddev = new MCvScalar();

            // Calculate the mean and standard deviation of the image
            CvInvoke.MeanStdDev(image, ref mean, ref stddev);

            // If the standard deviation is very high, the image is heavily noisy
            var noiseThreshold = 44.0; // Adjustable threshold setting 1
            var isHeavilyNoisy = stddev.V0 > noiseThreshold;

            results.Add(isHeavilyNoisy);
        }

        return results;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return null;
    }
}

/// <summary>
/// Determines if an image is tinted one colour, considering non-primary colours
/// </summary>
static IEnumerable<bool> IsTinted(string imagePath)
{
    try
    {
        var results = new List<bool>();

        // Load the image and split
        var imageQuadrants = SplitImageIntoQuadrants(imagePath, ImreadModes.Color);

        foreach (var image in imageQuadrants) 
        {
            // Split the image into its colour channels
            var channels = image.Split();

            // Calculate the mean and standard deviation of each channel
            var meanBlue = new MCvScalar();
            var stddevBlue = new MCvScalar();
            CvInvoke.MeanStdDev(channels[0], ref meanBlue, ref stddevBlue);

            var meanGreen = new MCvScalar();
            var stddevGreen = new MCvScalar();
            CvInvoke.MeanStdDev(channels[1], ref meanGreen, ref stddevGreen);

            var meanRed = new MCvScalar();
            var stddevRed = new MCvScalar();
            CvInvoke.MeanStdDev(channels[2], ref meanRed, ref stddevRed);

            // Calculate the differences between the means of the channels
            var diffBlueGreen = Math.Abs(meanBlue.V0 - meanGreen.V0);
            var diffBlueRed = Math.Abs(meanBlue.V0 - meanRed.V0);
            var diffGreenRed = Math.Abs(meanGreen.V0 - meanRed.V0);

            // Calculate the differences between the standard deviations of the channels
            var stddevDiffBlueGreen = Math.Abs(stddevBlue.V0 - stddevGreen.V0);
            var stddevDiffBlueRed = Math.Abs(stddevBlue.V0 - stddevRed.V0);
            var stddevDiffGreenRed = Math.Abs(stddevGreen.V0 - stddevRed.V0);

            // If one channel's mean or standard deviation is significantly different from the others, the image is tinted
            var tintThresholdMean = 50.0; // Adjustable threshold setting 2
            var tintThresholdStddev = 20.0; // Adjustable threshold setting 3
            var isTinted = diffBlueGreen > tintThresholdMean || diffBlueRed > tintThresholdMean || diffGreenRed > tintThresholdMean ||
                            stddevDiffBlueGreen > tintThresholdStddev || stddevDiffBlueRed > tintThresholdStddev || stddevDiffGreenRed > tintThresholdStddev;

            results.Add(isTinted);
        }
        return results;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return null;
    }
}

/// <summary>
/// Determines if an image is overexposed or underexposed
/// </summary>
static IEnumerable<string> AnalyzeExposure(string imagePath)
{
    var results = new List<string>();
    try
    {
        // Load the image and split
        var imageQuadrants = SplitImageIntoQuadrants(imagePath, ImreadModes.Color);

        foreach (var image in imageQuadrants)
        {
            UMat gray = new UMat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);
            VectorOfUMat vou = new VectorOfUMat();
            vou.Push(gray);

            // Calculate the histogram
            var histSize = new int[] { 256 };
            var ranges = new float[] { 0, 256 };

            var hist = new Mat();
            CvInvoke.CalcHist(vou, [0], null, hist, histSize, ranges, false);

            // Normalize the histogram
            CvInvoke.Normalize(hist, hist, 0, 1, NormType.MinMax);

            // Calculate the sum of the histogram values in the lower and upper ranges
            var lowerSum = 0.0;
            var upperSum = 0.0;
            var histData = (Single[,])hist.GetData();

            for (int i = 0; i < histData.GetLength(0); i++)
            {
                for (int j = 0; j < histData.GetLength(1); j++)
                {
                    var binValue = histData[i, j];
                    if (i < 50) // Lower range (dark pixels)
                    {
                        lowerSum += binValue;
                    }
                    else if (i > 200) // Upper range (bright pixels)
                    {
                        upperSum += binValue;
                    }
                }
            }

            // Determine if the image is overexposed or underexposed
            var exposureThreshold = 0.5; // Adjustable threshold setting
            if (upperSum > exposureThreshold)
            {
                results.Add("Overexposed");
            }
            else if (lowerSum > exposureThreshold)
            {
                results.Add("Underexposed");
            }
            else
            {
                results.Add("Properly Exposed");
            }
        }
        return results;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return null;
    }
}

/// <summary>
/// Splits an image into 4 quadrants
/// </summary>
static List<Mat> SplitImageIntoQuadrants(string imagePath, ImreadModes colourMode)
{
    try
    {
        // Load the image
        using var image = CvInvoke.Imread(imagePath, colourMode);

        // Get the dimensions of the image
        int width = image.Width;
        int height = image.Height;

        // Calculate the dimensions of each quadrant
        int halfWidth = width / 2;
        int halfHeight = height / 2;

        // Create a list to hold the quadrant images
        var quadrants = new List<Mat>();

        // Top-left quadrant
        var topLeft = new Mat(image, new Rectangle(0, 0, halfWidth, halfHeight));
        quadrants.Add(topLeft);

        // Top-right quadrant
        var topRight = new Mat(image, new Rectangle(halfWidth, 0, halfWidth, halfHeight));
        quadrants.Add(topRight);

        // Bottom-left quadrant
        var bottomLeft = new Mat(image, new Rectangle(0, halfHeight, halfWidth, halfHeight));
        quadrants.Add(bottomLeft);

        // Bottom-right quadrant
        var bottomRight = new Mat(image, new Rectangle(halfWidth, halfHeight, halfWidth, halfHeight));
        quadrants.Add(bottomRight);

        return quadrants;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message} {ex.Source}");
        return null;
    }
}
