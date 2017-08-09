function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Rasterization Demo';
	UI.titleShort = 'rasterizationDemo';

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Rasterization`,
		id: `RasterizationDemoFS`,
		initialValue: `#define PROJECTION
#define RASTERIZATION
#define CLIPPING
#define INTERPOLATION
//#define ZBUFFERING

precision highp float;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 VIEWPORT;

struct Vertex {
    vec3 position;
    vec3 color;
};



float magnitude(vec3 vector)
  {
    float mag = sqrt((vector.x*vector.x)+(vector.y*vector.y)+(vector.z*vector.z));
    return mag;
  }


struct Polygon {
    // Numbers of vertices, i.e., points in the polygon
    int vertexCount;
    // The vertices themselves
    Vertex vertices[MAX_VERTEX_COUNT];
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) {
            polygon.vertices[i] = element;
        }
    }
    polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        destination.vertices[i] = source.vertices[i];
    }
    destination.vertexCount = source.vertexCount;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) return polygon.vertices[0];
        if (i == index) return polygon.vertices[i];
    }
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
  polygon.vertexCount = 0;
}

// Clipping part

#define ENTERING 0
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {
#ifdef CLIPPING
    // Put your code here
  
   float e1 = (poli1.position.x-wind1.position.x)*(wind2.position.y-wind1.position.y)-(poli1.position.y-wind1.position.y)*(wind2.position.x-wind1.position.x);
  
float e2 = (poli2.position.x-wind1.position.x)*(wind2.position.y-wind1.position.y)-(poli2.position.y-wind1.position.y)*(wind2.position.x-wind1.position.x);
  
  if(e1>=0.0 && e2>=0.0)
    {
      return INSIDE;
    }
  if(e1<0.0 && e2<0.0)
    {
     	return OUTSIDE;
    }
   if(e1<0.0 && e2>=0.0)
    {
      return ENTERING;
    }
   if(e1>=0.0 && e2<0.0)
    {
      return LEAVING;
    } 
 
  
#else
    return INSIDE;
#endif
}



// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef CLIPPING
    // Put your code here
  
      float m1, m2, x, y, e, f;
      m1 = (b.position.y - a.position.y) / (b.position.x - a.position.x);
      e = a.position.y - a.position.x*m1;
  
      
      if (d.position.x - c.position.x != 0.0) {
          m2 = (d.position.y - c.position.y) / (d.position.x - c.position.x);
          f = c.position.y - c.position.x*m2;
          x = -(e-f)/(m1-m2);
        } else {
          x = c.position.x;
        }
  
      y = m1 * x + e;
  
      Vertex v;
      v.position = vec3(x, y, 0);
      v.color = vec3(1);
        return v;
    
#else
    return a;
#endif
}

void sutherlandHodgmanClip(Polygon unclipped, Polygon clipWindow, out Polygon result) {
    Polygon clipped;
    copyPolygon(clipped, unclipped);

    // Loop over the clip window
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i >= clipWindow.vertexCount) break;

        // Make a temporary copy of the current clipped polygon
        Polygon oldClipped;
        copyPolygon(oldClipped, clipped);

        // Set the clipped polygon to be empty
        makeEmptyPolygon(clipped);

        // Loop over the current clipped polygon
        for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
            if (j >= oldClipped.vertexCount) break;
            
            // Handle the j-th vertex of the clipped polygon. This should make use of the function 
            // intersect() to be implemented above.
#ifdef CLIPPING
            // Put your code here
          
         
         Vertex v1, v2, v3, v4, intersect;
          intersect.color=vec3(0.0);
          
       
          v3 = clipWindow.vertices[i];
          if(i+1==clipWindow.vertexCount)
            {
              v4 = clipWindow.vertices[0];
            }
          else
            {
              v4 =clipWindow.vertices[i+1];
            }
          
           v1 = oldClipped.vertices[j];
          if(j+1==oldClipped.vertexCount)
            {
              v2 = oldClipped.vertices[0];
            }
          else
            {
              v2 =oldClipped.vertices[j+1];
            }
          
          intersect = intersect2D(v1,v2, v3, v4);
          int cross = getCrossType(v1,v2,v3,v4);
           if(cross ==LEAVING)
            {
              
              appendVertexToPolygon(clipped, intersect);
            }else if(cross==ENTERING)
            {
              
               appendVertexToPolygon(clipped, intersect);
              appendVertexToPolygon(clipped, v2);
            }
            else if( cross ==INSIDE)
            {
              appendVertexToPolygon(clipped, v2);
            }

          
#else
            appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
        }

    
    }

    // Copy the last version to the output

    copyPolygon(result, clipped);
}

// Rasterization and culling part

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point 
// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {
#ifdef RASTERIZATION
    // Put your code here  
  
    float e = (point.x-a.position.x)*(b.position.y-a.position.y)-(point.y-a.position.y)*(b.position.x-a.position.x);
  
  if(e>=0.0)
    {
      return INNER_SIDE;
    }
  else
    {
      return OUTER_SIDE;
    }
  
#endif
    return OUTER_SIDE;
}

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
    // Don't evaluate empty polygons
    if (polygon.vertexCount == 0) return false;
    // Check against each edge of the polygon
    bool rasterise = true;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#ifdef RASTERIZATION
            // Put your code here

                    if(i!=(polygon.vertexCount-1))
                     {
                        if(edge(point, polygon.vertices[i], polygon.vertices[i+1])!=0)
                        {
                          rasterise=false;
                         return rasterise;
                        }
                    }
                    else
                    {
                        if(edge(point, polygon.vertices[i], polygon.vertices[0])!=0)
                        {
                         rasterise=false;
                         return rasterise;
                        }
                    }
                  
              
          
#else
            rasterise = false;
#endif
        }
    }
    return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
    float pointSize = 0.008;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            if(length(polygon.vertices[i].position.xy - point) < pointSize) return true;
        }
    }
    return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {
    // https://en.wikipedia.org/wiki/Heron%27s_formula
    float ab = length(a - b);
    float bc = length(b - c);
    float ca = length(c - a);
    float s = (ab + bc + ca) / 2.0;
    return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
    float weightSum = 0.0;
    vec3 colorSum = vec3(0.0);
    vec3 positionSum = vec3(0.0);
    float depthSum = 0.0;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#if defined(INTERPOLATION) || defined(ZBUFFERING)
            // Put your code here
         
#else
#endif
#ifdef ZBUFFERING
            // Put your code here
#endif
#ifdef INTERPOLATION
            // Put your code here
          
          //float k=i+1;
          //if(i==polygon.vertexCount-1) k=0;
          
          float area1, area2, area3, totalArea;
          vec2 p1, p2, p3;
          
          p1.x=polygon.vertices[0].position.x;
          p1.y=polygon.vertices[0].position.y;
          p2.x=polygon.vertices[1].position.x;
          p2.y=polygon.vertices[1].position.y;
          p3.x=polygon.vertices[2].position.x;
          p3.y=polygon.vertices[2].position.y;
          
          area1 = triangleArea(point, p2, p3);
          area2 = triangleArea(point, p3, p1);
          area3 = triangleArea(point, p1, p2);
          
          totalArea = triangleArea(p1, p2, p3);

          float b1,b2,b3;
          
          b1=area1/totalArea;
          b2=area2/totalArea;
          b3=area3/totalArea;
          
         colorSum = polygon.vertices[0].color*b1 + polygon.vertices[1].color*b2 + polygon.vertices[2].color*b3;
          
          
#endif
        }
    }
    
    Vertex result = polygon.vertices[0];
  
#ifdef INTERPOLATION
   // colorSum /= weightSum;
   // positionSum /= weightSum;
   // depthSum /= weightSum;
  //  colorSum /= depthSum;    
    result.color = colorSum;
#endif
#ifdef ZBUFFERING
    positionSum /= depthSum;
    result.position = positionSum;
#endif

  return result;
}

// Projection part

// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
    mat4 projectionMatrix = mat4(1);

#ifdef PROJECTION
    // Put your code here
  
   float d = 1.0;
  float capD = 9.0;
    
  mat4 pMatrix = mat4(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, -d,
  0, 0, 0, 1);
  
  mat4 scale = mat4(
  (1.0/capD), 0, 0, 0,
  0, (1.0/capD), 0, 0,
  0, 0, (1.0/capD), 0,
  0, 0, 0, 1);
  
  return scale*pMatrix;
  
#endif
  
    return projectionMatrix;
}

// Used to generate a simple "look-at" camera. 
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    mat4 viewMatrix = mat4(1);

#ifdef PROJECTION
    // Put your code here
  
  vec3 VPN = TP-VRP;
//  vec3 n = VPN/(magnitude(VPN));
  //vec3 u = (cross(n,VUV))/magnitude(cross(n,VUV));
  vec3 n = VPN/magnitude(VPN);
  vec3 u = cross(VUV, n)/magnitude(cross(VUV, n));
  vec3 v = cross(n,u);
  
  vec3 q = VRP;
  
  float t1, t2, t3;

  	t1 = -1.0*dot(u,q);
  	t2 = -1.0*dot(v,q);
  	t3 = -1.0*dot(n,q);
  
  mat4 m = mat4(
    u.x, u.y, u.z, t1,
    v.x, v.y, v.z, t2,
    n.x, n.y, n.z, t3,
    0, 0, 0, 1);
  
  return m;
  
  
#endif
    return viewMatrix;
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec3 projectVertexPosition(vec3 position) {

  // Set the parameters for the look-at camera.
    vec3 TP = vec3(0, 0, 0);
    vec3 VRP = vec3(0, 0, -7);
    vec3 VUV = vec3(0, 1, 0);
  
    // Compute the view matrix.
    mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  // Compute the projection matrix.
    mat4 projectionMatrix = computeProjectionMatrix();
  
#ifdef PROJECTION
    // Put your code here
  
    vec4 a;
  a[0]=position[0];
  a[1]=position[1];
  a[2]=position[2];
  a[3]=1.0;
  
 // vec4 b = projectionMatrix*viewMatrix*a;
  vec4 b = a*viewMatrix*projectionMatrix;
  
  vec3 c;
  c[0]=b[0]/b[3];
  c[1]=b[1]/b[3];
  c[2]=b[2]/b[3];
  
  return c;
  
#else
    return position;
#endif
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
    copyPolygon(projectedPolygon, polygon);
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
        }
    }
}

// Draws a polygon by projecting, clipping, ratserizing and interpolating it
void drawPolygon(
  vec2 point, 
  Polygon clipWindow, 
  Polygon oldPolygon, 
  inout vec3 color, 
  inout float depth)
{
    Polygon projectedPolygon;
    projectPolygon(projectedPolygon, oldPolygon);  
  
    Polygon clippedPolygon;
    sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

    if (isPointInPolygon(point, clippedPolygon)) {
      
        Vertex interpolatedVertex = 
          interpolateVertex(point, projectedPolygon);
          
        if (interpolatedVertex.position.z < depth) {
            color = interpolatedVertex.color;
            depth = interpolatedVertex.position.z;
        }
    } else {
        if (isPointInPolygon(point, projectedPolygon)) {
            color = vec3(0.1, 0.1, 0.1);
        }
    }
  
   if (isPointOnPolygonVertex(point, clippedPolygon)) {
        color = vec3(1);
   }
}

// Main function calls

void drawScene(vec2 point, inout vec3 color) {
    color = vec3(0.3, 0.3, 0.3);
    point = vec2(2.0 * point.x / float(VIEWPORT.x) - 1.0, 2.0 * point.y / float(VIEWPORT.y) - 1.0);

    Polygon clipWindow;
    clipWindow.vertices[0].position = vec3(-0.750,  0.750, 1.0);
    clipWindow.vertices[1].position = vec3( 0.750,  0.750, 1.0);
    clipWindow.vertices[2].position = vec3( 0.750, -0.750, 1.0);
    clipWindow.vertices[3].position = vec3(-0.750, -0.750, 1.0);
    clipWindow.vertexCount = 4;
    color = isPointInPolygon(point, clipWindow) ? vec3(0.5, 0.5, 0.5) : color;

    const int triangleCount = 2;
    Polygon triangles[triangleCount];
  
    triangles[0].vertices[0].position = vec3(-7.7143, -3.8571, 1.0);
    triangles[0].vertices[1].position = vec3(7.7143, 8.4857, 1.0);
    triangles[0].vertices[2].position = vec3(4.8857, -0.5143, 1.0);
    triangles[0].vertices[0].color = vec3(1.0, 0.5, 0.1);
    triangles[0].vertices[1].color = vec3(0.2, 0.8, 0.2);
    triangles[0].vertices[2].color = vec3(0.2, 0.3, 1.0);
    triangles[0].vertexCount = 3;
  
    triangles[1].vertices[0].position = vec3(3.0836, -4.3820, 1.9);
    triangles[1].vertices[1].position = vec3(-3.9667, 0.7933, 0.5);
    triangles[1].vertices[2].position = vec3(-4.3714, 8.2286, 1.0);
    triangles[1].vertices[1].color = vec3(0.1, 0.5, 1.0);
    triangles[1].vertices[2].color = vec3(1.0, 0.6, 0.1);
    triangles[1].vertices[0].color = vec3(0.2, 0.6, 1.0);
    triangles[1].vertexCount = 3;

    float depth = 10000.0;
    // Project and draw all the triangles
    for (int i = 0; i < triangleCount; i++) {
        drawPolygon(point, clipWindow, triangles[i], color, depth);
    }   
}

void main() {
    drawScene(gl_FragCoord.xy, gl_FragColor.rgb);
    gl_FragColor.a = 1.0;
}`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: true,
		type: `text/javascript`,
		title: `Resolution settings`,
		id: `ResolutionJS`,
		initialValue: `// This variable sets the inverse scaling factor at which the rendering happens.
// The higher the constant, the faster it will be. SCALING = 1 is regular, non-scaled rendering.
SCALING = 1;`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoTextureVS - GL`,
		id: `RasterizationDemoTextureVS`,
		initialValue: `attribute vec3 position;
    attribute vec2 textureCoord;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    varying highp vec2 vTextureCoord;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        vTextureCoord = textureCoord;
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoVS - GL`,
		id: `RasterizationDemoVS`,
		initialValue: `attribute vec3 position;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-fragment`,
		title: `RasterizationDemoTextureFS - GL`,
		id: `RasterizationDemoTextureFS`,
		initialValue: `
        varying highp vec2 vTextureCoord;

        uniform sampler2D uSampler;

        void main(void) {
            gl_FragColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
        }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
    try {
        gl = canvas.getContext("webgl");
        gl.viewportWidth = canvas.width;
        gl.viewportHeight = canvas.height;
    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(");
    }
}

function evalJS(id) {
    var jsScript = document.getElementById(id);
    eval(jsScript.innerHTML);
}

function getShader(gl, id) {
    var shaderScript = document.getElementById(id);
    if (!shaderScript) {
        return null;
    }

    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) {
            str += k.textContent;
        }
        k = k.nextSibling;
    }

    var shader;
    if (shaderScript.type == "x-shader/x-fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } else if (shaderScript.type == "x-shader/x-vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function RasterizationDemo() {
}

RasterizationDemo.prototype.initShaders = function() {

    this.shaderProgram = gl.createProgram();

    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoVS"));
    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoFS"));
    gl.linkProgram(this.shaderProgram);

    if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.shaderProgram);

    this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
    gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

    this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
    this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initTextureShaders = function() {

    this.textureShaderProgram = gl.createProgram();

    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureVS"));
    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureFS"));
    gl.linkProgram(this.textureShaderProgram);

    if (!gl.getProgramParameter(this.textureShaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.textureShaderProgram);

    this.textureShaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.textureShaderProgram, "position");
    gl.enableVertexAttribArray(this.textureShaderProgram.vertexPositionAttribute);

    this.textureShaderProgram.textureCoordAttribute = gl.getAttribLocation(this.textureShaderProgram, "textureCoord");
    gl.enableVertexAttribArray(this.textureShaderProgram.textureCoordAttribute);
    //gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, 2, gl.FLOAT, false, 0, 0);

    this.textureShaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "projectionMatrix");
    this.textureShaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initBuffers = function() {
    this.triangleVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
    
    var vertices = [
         -1.0,  -1.0,  0.0,
         -1.0,   1.0,  0.0,
          1.0,   1.0,  0.0,

         -1.0,  -1.0,  0.0,
          1.0,  -1.0,  0.0,
          1.0,   1.0,  0.0,
     ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    this.triangleVertexPositionBuffer.itemSize = 3;
    this.triangleVertexPositionBuffer.numItems = 3 * 2;

    this.textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);

    var textureCoords = [
        0.0,  0.0,
        0.0,  1.0,
        1.0,  1.0,

        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoords), gl.STATIC_DRAW);
    this.textureCoordBuffer.itemSize = 2;
}

RasterizationDemo.prototype.initTextureFramebuffer = function() {
    // create off-screen framebuffer
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    this.framebuffer.width = this.prerender_width;
    this.framebuffer.height = this.prerender_height;

    // create RGB texture
    this.framebufferTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.framebuffer.width, this.framebuffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);//LINEAR_MIPMAP_NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    //gl.generateMipmap(gl.TEXTURE_2D);

    // create depth buffer
    this.renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.framebuffer.width, this.framebuffer.height);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.framebufferTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.renderbuffer);

    // reset state
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

RasterizationDemo.prototype.drawScene = function() {
            
    gl.bindFramebuffer(gl.FRAMEBUFFER, env.framebuffer);
    gl.useProgram(this.shaderProgram);
    gl.viewport(0, 0, this.prerender_width, this.prerender_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);

        gl.uniform2iv(gl.getUniformLocation(this.shaderProgram, "VIEWPORT"), [this.prerender_width, this.prerender_height]);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.textureShaderProgram);
    gl.viewport(0, 0, this.render_width, this.render_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.textureShaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.textureShaderProgram.modelviewMatrixUniform, false);

        gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
        gl.uniform1i(gl.getUniformLocation(this.textureShaderProgram, "uSampler"), 0);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RasterizationDemo.prototype.run = function() {
    evalJS("ResolutionJS");

    this.render_width     = 800;
    this.render_height    = 400;

    this.prerender_width  = this.render_width / SCALING;
    this.prerender_height = this.render_height / SCALING;

    this.initTextureFramebuffer();
    this.initShaders();
    this.initTextureShaders();
    this.initBuffers();
};

function init() {   
    env = new RasterizationDemo();

    return env;
}

function compute(canvas)
{
    env.run();
    env.drawScene();
}
