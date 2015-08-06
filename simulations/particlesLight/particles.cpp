/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Particle system example with collisions using uniform grid
    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.
    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

#define ENABLE_DEBUG_OUTPUT
#ifdef ENABLE_DEBUG_OUTPUT
#define debug(s, ...)                                           \
  do {                                                          \
    fprintf (stderr, "(%-30s:%40s:%4d) -- " s,                  \
             __FILE__, __func__, __LINE__, ##__VA_ARGS__);      \
    fflush (stderr);                                            \
  } while (0)
#else
#define debug(s, ...)
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>
#include <sys/time.h>

#include "particleSystem.h"
#include "voxelObject.h"
#include "render_particles.h"
#include "paramgl.h"
#include "event_timer.h"

// TODO: remove
#include "gpuVoxelTree.h"

// Parameters you might be interested in changing (also command line)
uint numParticles = 180424;
uint3 gridSize = {256, 256, 256};
int numIterations = 1000; // run until exit

bool usingObject = false;
bool usingSpout = false;
bool limitLifeByHeight = false;
bool limitLifeByTime = false;

// simulation parameters
float timestep = 0.1f;
float damping = 1.0f;
float gravity = 0.0003f;
int ballr = 10;

float collideSpring = 0.5f;;
float collideDamping = 0.02f;;
float collideShear = 0.1f;
float collideAttraction = 0.0f;

const uint width = 640*2, height = 480*2;

// view params
int ox, oy;
int buttonState = 0;
#if 0
// these are debugging values for the camera location
float camera_trans[] = {0, 0, -1.5};
float camera_rot[]   = {-3.6, -73.4, 0};
#else
float camera_trans[] = {0, 0, -15};
float camera_rot[]   = {0, 0, 0};
const float inertia = 0.1f;
//ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_POINTS;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

int mode = 0;
bool displayEnabled = true;
bool pauseSpout = false;
bool moveSpout = true;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;

// For keeping frame rate consistent
struct timeval timeOfLastPhysics;
struct timeval timeOfLastRender; 

enum { M_VIEW = 0, M_MOVE };

ParticleSystem *psystem = 0;
VoxelTree *voxelTree = 0; 

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

void
writeTimes(const float* times,
           const std::string& filename,
           const uint numParticles,
           const uint gridSize) {
    const std::string appendedFilename = filename + std::string(".csv");
    FILE* file = fopen(appendedFilename.c_str(), "a");
    fprintf(file, "%d, ", numParticles);
    for (unsigned int i = 0; i < 6; ++i) {
        fprintf(file, "%f", times[i]);
        if (i != 5) {
            fprintf(file, ",");
        }
    }
    fprintf(file, "\n");
    fclose(file);
    printf("wrote file to %s\n", appendedFilename.c_str());
}

void calcNumNeighbors(const uint* neighbors, uint* neighborStats, const uint numParticles, const uint maxNeighbors) {
    for (int i=1; i < numParticles + 1; ++i) {
        int numNeighbors = neighbors[i];
        if (numNeighbors < maxNeighbors) {
            neighborStats[numNeighbors] += 1;
        }
        else {
            printf("Neighbors is %d. Total is %d. Index is %d\n", numNeighbors, psystem->getNumActiveParticles(), i);
        }
    }

}

void
writeNeighbors(const uint* neighbors,
           const std::string& filename,
           const uint numParticles,
           const uint maxNeighbors) {
    const std::string appendedFilename = filename + std::string(".csv");
    uint* neighborStats = new uint[maxNeighbors]; 
    memset(neighborStats, 0, maxNeighbors*sizeof(uint)); 
    calcNumNeighbors(neighbors, neighborStats, numParticles, maxNeighbors);
    FILE* file = fopen(appendedFilename.c_str(), "a");
    if (file == NULL) {
        printf("Error opening file\n");
    }
    fprintf(file, "%d, ", numParticles);
    fprintf(file, "%d, ", neighbors[0]);
    for (int i=0; i < maxNeighbors; ++i) {
        fprintf(file, "%d", neighborStats[i]);
        if (i != maxNeighbors - 1) {
            fprintf(file, ", ");
        }
    }
    fprintf(file, "\n");
    fclose(file);
    delete [] neighborStats;
    //printf("wrote file to %s\n", appendedFilename.c_str());
}

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL);
    ParticleSystem::ParticleConfig config = ParticleSystem::CONFIG_GRID;
    if (usingSpout) {
        config = ParticleSystem::CONFIG_SPOUT;
    }
    psystem->reset(config);
    psystem->startTimer(5);

    unsigned int blah[4] = {4, 4, 4, 4};
    
    std::vector<unsigned int> cellsPerSide(blah, blah + sizeof(blah) / sizeof(blah[0]));

    voxelTree = new VoxelTree(cellsPerSide);
    voxelTree->initializeTree();
    voxelTree->initializeShape();

    size_t free_byte ;

    size_t total_byte ;

    checkCudaErrors(cudaMemGetInfo( &free_byte, &total_byte )) ;



    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

    used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    if (bUseOpenGL)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
    }

    gettimeofday(&timeOfLastPhysics, 0);
    gettimeofday(&timeOfLastRender, 0);

}

static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // good old-fashioned fixed function lighting
    float black[]    = { 0.0f, 0.0f, 0.0f, 1.0f };
    float white[]    = { 1.0f, 1.0f, 1.0f, 1.0f };
    float ambient[]  = { 0.1f, 0.1f, 0.1f, 1.0f };
    float diffuse[]  = { 0.7f, 0.7f, 0.7f, 1.0f };
    float lightPos[] = { 0.0f, 0.0f, 0.0f, 0.0f };

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);

    GLint gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

    glutReportErrors();
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

void getColor(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void display()
{
    // update the simulation
    struct timeval currentTime;
    gettimeofday(&currentTime, 0);
    double timeDiff = (double)(1000.0 * (currentTime.tv_sec - timeOfLastPhysics.tv_sec)
                   + (0.001 * (currentTime.tv_usec - timeOfLastPhysics.tv_usec)));
    if (timeDiff >= (timestep/0.5) * 10.0)
    {
        timeOfLastPhysics = currentTime;
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);
        psystem->setRotation(camera_rot);
        psystem->setTranslation(camera_trans);

        const unsigned int timestepIndex = frameCount;
        psystem->update(timestep, timestepIndex, voxelObject, pauseSpout, moveSpout);

        if (renderer)
        {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumActiveParticles());
        }
    }
    

    // Do the rendering
    gettimeofday(&currentTime, 0);
    timeDiff = (double)(1000.0 * (currentTime.tv_sec - timeOfLastRender.tv_sec)
                   + (0.001 * (currentTime.tv_usec - timeOfLastRender.tv_usec)));
    if (timeDiff >= (1.0/60.0) * 1000.0) {
        timeOfLastRender = currentTime;
        // render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // view transform
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        
        for (int c = 0; c < 3; ++c)
        {
            camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
            camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
        }
    
#if 0
        glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
        glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
        glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
#else
        glTranslatef(camera_trans[0], camera_trans[1], camera_trans[2]);
        glRotatef(camera_rot[0], 1.0, 0.0, 0.0);
        glRotatef(camera_rot[1], 0.0, 1.0, 0.0);
#endif
#if 0
    printf("camera = (%5.1f %5.1f %5.1f) (%5.1f %5.1f)\n",
           camera_trans[0],
           camera_trans[1],
           camera_trans[2],
           camera_rot[0],
           camera_rot[1]);
#endif

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(8.0);


    // TODO: Remove
    // testing the VoxelTree
    // VoxelTree::test();

    // collider
    /*
    glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(1.0, 0.0, 0.0);
    glutSolidSphere(psystem->getColliderRadius(), 20, 10);
    glPopMatrix();
    */
    if (renderer && displayEnabled)
    {
        renderer->display(displayMode);
    }
    // voxelTree->renderVoxelTree(modelView, psystem->getParticleRadius()); 
    voxelTree->debugDisplay();

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(8.0);

        if (renderer && displayEnabled)
        {
            renderer->setColorBuffer(psystem->getColorBuffer());
            renderer->setParticleRadius(psystem->getParticleRadius());
            renderer->setPointSize(psystem->getParticleRadius());
            renderer->display(displayMode);
        }

#if 0
        if (renderer)
          {
            renderer->setColorBuffer(voxelObject->getColorBuffer());
            renderer->setVertexBuffer(voxelObject->getCurrentReadBuffer(), voxelObject->getNumVoxels());
            renderer->setParticleRadius(voxelObject->getVoxelSize());
            renderer->setPointSize(50 * voxelObject->getVoxelSize());
            renderer->display(displayMode);
          }
#endif

        /*
        if (displaySliders)
        {
            glDisable(GL_DEPTH_TEST);
            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
            glEnable(GL_BLEND);
            params->Render(0, 0);
            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
        }
        */
        

        glutSwapBuffers();
        glutReportErrors();
        ++frameCount;
    }

    // Keep track of frames to calculate FPS at end 

    //writeNeighbors(psystem->getNumNeighbors(), "numNeighbors", numParticles, 150);

    if (frameCount >=numIterations) {
#ifndef __APPLE__
      glutLeaveMainLoop();
#endif
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void addSphere()
{
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr+(pr*2.0f)*ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr*2.0f);
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer)
    {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders)
    {
        if (params->Mouse(x, y, button, state))
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
    r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
    r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
    r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
    r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode)
    {
        case M_VIEW:
            if (buttonState == 3)
            {
                // left+middle = zoom
                camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
            }
            else if (buttonState & 2)
            {
                // middle = translate
                camera_trans[0] += dx / 100.0f;
                camera_trans[1] -= dy / 100.0f;
            }
            else if (buttonState & 1)
            {
                // left = rotate
                camera_rot[0] += dy / 5.0f;
                camera_rot[1] += dx / 5.0f;
            }

            break;

        case M_MOVE:
            {
                float translateSpeed = 0.003f;
                float3 p = psystem->getColliderPos();

                if (buttonState==1)
                {
                    float v[3], r[3];
                    v[0] = dx*translateSpeed;
                    v[1] = -dy*translateSpeed;
                    v[2] = 0.0f;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }
                else if (buttonState==2)
                {
                    float v[3], r[3];
                    v[0] = 0.0f;
                    v[1] = 0.0f;
                    v[2] = dy*translateSpeed;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }

                psystem->setColliderPos(p);
            }
            break;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    const unsigned int timestepIndex = frameCount;
    switch (key)
    {
        /*case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(timestep, voxelTree);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
            }

            break;*/

        case '\033':
        case 'q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case 'm':
            moveSpout = !moveSpout;
            break;

        case 'p':
            pauseSpout = !pauseSpout;
            break;

        case 'd':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;

        /*case 'd':
            psystem->dumpGrid();
            break;

        case 'u':
            psystem->dumpParticles(0, numParticles-1);
            break;

        case 'r':
            displayEnabled = !displayEnabled;
            break;

        case '1':
            psystem->reset(ParticleSystem::CONFIG_GRID);
            break;

        case '2':
            psystem->reset(ParticleSystem::CONFIG_RANDOM);
            break;

        case '3':
            addSphere();
            break;

        case '4':
            {
                // shoot ball from camera
                float pr = psystem->getParticleRadius();
                float vel[4], velw[4], pos[4], posw[4];
                vel[0] = 0.0f;
                vel[1] = 0.0f;
                vel[2] = -0.05f;
                vel[3] = 0.0f;
                ixform(vel, velw, modelView);

                pos[0] = 0.0f;
                pos[1] = 0.0f;
                pos[2] = -2.5f;
                pos[3] = 1.0;
                ixformPoint(pos, posw, modelView);
                posw[3] = 0.0f;

                psystem->addSphere(0, posw, velw, ballr, pr*2.0f);
            }
            break;

        case 'w':
            wireframe = !wireframe;
            break;

        case 'h':
            displaySliders = !displaySliders;
            break; */
    }

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders)
    {
        params->Special(k, x, y);
    }

    demoMode = false;
    idleCounter = 0;
}

void idle(void)
{
    /*if ((idleCounter++ > idleDelay) && (demoMode==false))
    {
        demoMode = true;
        printf("Entering demo mode\n");
    }
    if (demoMode)
    {
        camera_rot[1] += 0.1f;
        if (demoCounter++ > 1000)
        {
            ballr = 10 + (rand() % 10);
            addSphere();
            demoCounter = 0;
        }
    }
    */
    glutPostRedisplay();
}

void initParams()
{
    // create a new parameter list
    params = new ParamListGL("misc");
    params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
    params->AddParam(new Param<float>("damping"  , damping , 0.0f, 1.0f, 0.001f, &damping));
    params->AddParam(new Param<float>("gravity"  , gravity , 0.0f, 0.001f, 0.0001f, &gravity));
    params->AddParam(new Param<int> ("ball radius", ballr , 1, 20, 1, &ballr));

    params->AddParam(new Param<float>("collide spring" , collideSpring , 0.0f, 1.0f, 0.001f, &collideSpring));
    params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
    params->AddParam(new Param<float>("collide shear"  , collideShear  , 0.0f, 0.1f, 0.001f, &collideShear));
    params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKsample);

    printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
        {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "grid"))
        {
            gridSize.x=gridSize.y = gridSize.z = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
        }
        if (checkCmdLineFlag(argc, (const char **) argv, "time"))
        {
            timestep = getCmdLineArgumentFloat(argc, (const char **) argv, "time");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "i"))
        {
            numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
        }
        if (checkCmdLineFlag(argc, (const char **) argv, "-o"))
        {
            usingObject = true;
        }
        if (checkCmdLineFlag(argc, (const char **) argv, "-s"))
        {
            usingSpout = true;
        }
        if (checkCmdLineFlag(argc, (const char **) argv, "-h"))
        {
            limitLifeByHeight = true;
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "-t"))
        {
            limitLifeByTime = true;
        }

    }

    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("particles: %d\n", numParticles);
    printf("Usage: Press m to move or stop moving the spout with the camera.\n");
    printf("Press p to pause or unpause particles coming out of the spout.\n");


    initGL(&argc, argv);
    cudaGLInit(argc, argv);

    // Last param for initParticleSystem tells it to use openGL
    initParticleSystem(numParticles, gridSize, 1);
    initParams();
    
    initMenus();

#ifndef __APPLE__
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

    //glutCloseFunc(cleanup);
    glutMainLoop();

    psystem->stopTimer(5);
    float* times = psystem->getTime();
    writeTimes(times, "broadcastTimings", numParticles, gridSize.x);

    cudaThreadSynchronize();
 
     // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    // Note: cudaDeviceReset() cause a segfault. Not completely sure why
    // If you need profiling information, you'll have to fix the segfault.
    // Sorry :(
    // cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
