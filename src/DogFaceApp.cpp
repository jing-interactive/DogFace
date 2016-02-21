#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Text.h"
#include "cinder/gl/TextureFont.h"

// ocv
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "CinderOpenCV.h"
#include "CaptureHelper.h"

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace cv;

/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    cv::Point classNumber;
    
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

std::vector<string> readClassNames(const char *filename)
{
    std::vector<string> classNames;
    
    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        console() << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }
    
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }
    
    fp.close();
    return classNames;
}

class DogFaceApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
    
private:
    bool getForwardOutput(dnn::Blob& outputBlob);
    CaptureHelper mCaptureHelper;
    dnn::Net net;
    vector<string> mClassNames;
    dnn::Blob mNetOutput;
    
    shared_ptr<thread> mForwardThread;
    string mInformation = "WOW! I am loading.";
    
    Font				mFont;
    gl::TextureFontRef	mTextureFont;
};

void DogFaceApp::setup()
{
    mCaptureHelper.setup();
    
    // iOS 6: Font list
    // https://support.apple.com/en-us/HT202599
#if defined( CINDER_COCOA_TOUCH )
    mFont = Font( "Helvetica", 20 );
#elif defined( CINDER_COCOA )
    mFont = Font( "BigCaslon-Medium", 20 );
#else
    mFont = Font( "Times New Roman", 20 );
#endif
    mTextureFont = gl::TextureFont::create( mFont );
    
    auto forwardFn = [this]
    {
        String modelTxt = getAssetPath("bvlc_googlenet.prototxt").c_str();
        String modelBin = getAssetPath("bvlc_googlenet.caffemodel").c_str();
        
        //! [Create the importer of Caffe model]
        Ptr<dnn::Importer> importer;
        try                                     //Try to import Caffe GoogleNet model
        {
            importer = dnn::createCaffeImporter(modelTxt, modelBin);
        }
        catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
        {
            console() << err.msg << std::endl;
        }
        //! [Create the importer of Caffe model]
        
        if (!importer)
        {
            console() << "Can't load network by using the following files: " << std::endl;
            console() << "prototxt:   " << modelTxt << std::endl;
            console() << "caffemodel: " << modelBin << std::endl;
            console() << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            console() << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }
        
        //! [Initialize network]
        importer->populateNet(net);
        importer.release();                     //We don't need importer anymore
        //! [Initialize network]
        const char* labelTxt = getAssetPath("synset_words.txt").c_str();
        mClassNames = readClassNames(labelTxt);
        
        while (true)
        {
            auto startSeconds = getElapsedSeconds();
            auto pass = getForwardOutput(mNetOutput);
            if (!pass) continue;
            
            auto updateScoreFn = [this, startSeconds]
            {
                int classId;
                double classProb;
                ::getMaxClass(mNetOutput, &classId, &classProb);//find the best class
                //! [Gather output]
                
                auto forwardCostSeconds = getElapsedSeconds() - startSeconds;

                //! [Print results]
                console() << "Best class: #" << classId << " '" << mClassNames.at(classId) << "'" << std::endl;
                console() << "Probability: " << classProb * 100 << "%" << std::endl;
                char info[256];
                sprintf(info, "Doge! It took me %.1f seconds.\n%s", forwardCostSeconds, mClassNames.at(classId).c_str());
                mInformation = info;
                //! [Print results]
            };
            dispatchAsync(updateScoreFn);
        }
    };
    mForwardThread = shared_ptr<thread>( new thread( forwardFn ) );
}

bool DogFaceApp::getForwardOutput(dnn::Blob& outputProb)
{
    //! [Prepare blob]
#if 0
    String imageFile = getAssetPath("space_shuttle.jpg").c_str();
    Mat img = imread(imageFile);
    if (img.empty())
    {
        console() << "Can't read image from the file: " << imageFile << std::endl;
        return false;
    }
#else
    Mat imgRGBA = toOcv(mCaptureHelper.surface);
    if (imgRGBA.empty())
    {
        console() << "Video capture is not ready" << std::endl;        
        return false;
    }
    Mat img;
    cv::cvtColor(imgRGBA, img, COLOR_RGBA2RGB);
#endif
    cv::resize(img, img, cv::Size(224, 224));       //GoogLeNet accepts only 224x224 RGB-images
    dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]
    
    //! [Set input blob]
    net.setBlob(".data", inputBlob);        //set the network input
    //! [Set input blob]
    
    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]
    
    //! [Gather output]
    outputProb = net.getBlob("prob");   //gather output of "prob" layer
    
    return true;
}

void DogFaceApp::mouseDown( MouseEvent event )
{
}

void DogFaceApp::update()
{

}

void DogFaceApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
    
    gl::setMatricesWindow(getWindowSize());

    if( mCaptureHelper.texture ) {
        gl::ScopedModelMatrix modelScope;
        
#if defined( CINDER_COCOA_TOUCH )
        // change iphone to landscape orientation
        gl::rotate( M_PI / 2 );
        gl::translate( 0, - getWindowWidth() );
        
        Rectf flippedBounds( 0, 0, mCaptureHelper.size.x, mCaptureHelper.size.y );
        gl::draw( mCaptureHelper.texture, flippedBounds );
#else
        gl::draw( mCaptureHelper.texture );
#endif
    }
    
    Rectf boundsRect( 40, mTextureFont->getAscent() + 40, getWindowWidth() - 40, getWindowHeight() - 40 );
    
    gl::ScopedColor fontColor(ColorA( 0.8f, 1.0f, 0.75f, 1.0f ));
    mTextureFont->drawStringWrapped( mInformation, boundsRect );
}

CINDER_APP( DogFaceApp, RendererGl )
