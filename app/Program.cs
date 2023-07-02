using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class License_Plate_Recognition
{
    // The main entry point for the application.
    static void Main(string[] args)
    {
        // Load the Image file
        /* If u want to chage the image just
           change the path between Double Quotes("").*/
        Mat frame = CvInvoke.Imread(@"D:\app\test.jpg");

        // Perform license plate recognition on the frame
        ProcessFrame(frame);
    }

    // This function displays the frame.
    static void display_Picture(Mat frame)
    {
        CvInvoke.Imshow("License Plate", frame);
        CvInvoke.WaitKey(0);
        //closing all open windows
        CvInvoke.DestroyAllWindows();
    }

    // To apply the Gaussian Blur.
    static Mat blur_Img(Mat grayImage)
    {
        CvInvoke.GaussianBlur(grayImage, grayImage, new System.Drawing.Size(5, 5), 0);
        return grayImage;
    }

    // This is to apply Edge Detection Sobel Filter.
    static Mat sobel_Filter(Mat grayImage)
    {
        Mat edges = new Mat();
        CvInvoke.Sobel(grayImage, edges, DepthType.Cv8U, 1, 0, 3, 1, 0, BorderType.Default);
        return edges;
    }

    // This is to apply Threshold to binarize the image.
    static Mat threshold_Img(Mat edges)
    {
        CvInvoke.Threshold(edges, edges, 0, 255, ThresholdType.Otsu);
        return edges;
    }

    // This is to apply Morphological closing on picture.
    static Mat morphological_Closing(Mat edges)
    {
        Mat closed = new Mat();
        Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(17, 3), new System.Drawing.Point(-1, -1));
        CvInvoke.MorphologyEx(edges, closed, MorphOp.Close, kernel, new System.Drawing.Point(-1, -1), 3, BorderType.Default, new MCvScalar(255, 255, 255));
        return closed;
    }

    static void ProcessFrame(Mat frame)
    {
        // Step 1: Preprocessing
        Mat grayImage = new Mat();
        CvInvoke.CvtColor(frame, grayImage, ColorConversion.Bgr2Gray);
        grayImage = blur_Img(grayImage);

        // Step 2: Edge detection
        Mat edges = new Mat();
        edges = sobel_Filter(grayImage);
        edges = threshold_Img(edges);
        
        // Step 3: Morphological closing
        Mat closed = new Mat();
        closed = morphological_Closing(edges);

        // Step 4: Contour detection
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(closed, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        // Step 5: License plate validation
        List<System.Drawing.Rectangle> licensePlateRectangles = new List<System.Drawing.Rectangle>();
        for (int i = 0; i < contours.Size; i++)
        {
            VectorOfPoint contour = contours[i];
            System.Drawing.Rectangle boundingRect = CvInvoke.BoundingRectangle(contour);

            double aspectRatio = (double)boundingRect.Width / boundingRect.Height;
            double area = CvInvoke.ContourArea(contour);

            if (aspectRatio > 2.5 && aspectRatio < 7 && area > 4000 && area < 30000)
            {
                licensePlateRectangles.Add(boundingRect);
            }
        }

        // Step 6: Character segmentation and recognition
        foreach (System.Drawing.Rectangle plateRect in licensePlateRectangles)
        {
            // Extract license plate region
            Mat plateImage = new Mat(frame, plateRect);

            // Convert to grayscale
            Mat plateGray = new Mat();
            CvInvoke.CvtColor(plateImage, plateGray, ColorConversion.Bgr2Gray);

            // Binarize using adaptive thresholding
            Mat plateThreshold = new Mat();
            CvInvoke.AdaptiveThreshold(plateGray, plateThreshold, 255, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, 4);

            // Character candidates extraction
            Mat plateCandidates = new Mat();
            CvInvoke.BitwiseNot(plateThreshold, plateCandidates);

            // Find contours of character candidates
            VectorOfVectorOfPoint candidateContours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(plateCandidates, candidateContours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            List<System.Drawing.Rectangle> characterRectangles = new List<System.Drawing.Rectangle>();
            for (int i = 0; i < candidateContours.Size; i++)
            {
                VectorOfPoint candidateContour = candidateContours[i];
                System.Drawing.Rectangle candidateRect = CvInvoke.BoundingRectangle(candidateContour);

                double aspectRatio = (double)candidateRect.Width / candidateRect.Height;
                double area = CvInvoke.ContourArea(candidateContour);

                if (aspectRatio > 0.1 && aspectRatio < 1 && area > 100)
                {
                    characterRectangles.Add(candidateRect);
                }
            }

            // Extract characters from plate image
            foreach (System.Drawing.Rectangle characterRect in characterRectangles)
            {
                display_Picture(plateGray);
                return;
            }
        }
    }
}