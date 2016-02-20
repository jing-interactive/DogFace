#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class DogFaceApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void DogFaceApp::setup()
{
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
}

CINDER_APP( DogFaceApp, RendererGl )
