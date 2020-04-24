var context = new Object();

function clear() {
  context.canvas.clearRect(0,0,canvas.width,canvas.height);
} 

function drawImg(img) {
  context.canvas.drawImage(img, 0, 0, canvas.width, canvas.height);
}

function loadImg(path) {
  var image = new Image();
  context.cacheCount++;
  image.onload = function() {        
    context.cacheCount--;
    flush();
  };
  image.src = path;
  return image;
}

function lazyDraw() {
  if ("bgImg" in context) drawImg(context.bgImg);
  context.cachedImgs.forEach(drawImg);
  if (context.autoplay && "playImg" in context) drawImg(context.playImg);
}

function flush() {
  if (context.cacheCount==0) {
    lazyDraw();
    setTimeout( refreshEvent, context.timeout );
  }
}

function drawImgFromFile(path) {
  var image = new Image();
  image.onload = function() {        
    canvasCtx.drawImage(image, 0, 0, canvas.width, canvas.height);
  };
  image.src = path;
}

function keyEvent(e) {
  var key = e.code;
  if (key=='ArrowRight') next();
  if (key=='ArrowLeft') previous();
  if (key=='KeyP') togglePlay();
  if (key=='PageUp') start();
  if (key=='PageDown') end();        
}

function start() {
  context.index = 0;
  context.autoplay = false;
  update();
}

function end() {
  context.index = context.frames.length-1;
  context.autoplay = false;
  update();
}

function next() {
  if (context.index==(context.frames.length-1))
    context.autoplay = false;
  else context.index++;
  update();
}

function previous() {
  if (context.index==0) return;
  context.index--;
  update();
}

function update() {
  context.cachedImgs = [];
  var i;
  for (i = 0; i<context.frames[context.index].length; i++) {
    context.cachedImgs[i] = loadImg(context.frames[context.index][i]);
  }
  flush();
}

function togglePlay() {
  context.autoplay = !context.autoplay;
  update();
}

function refreshEvent() {
  if (context.autoplay) next();
  else flush();
}

function addLayer(path,fromIdx, toIdx=null) {
  if (toIdx==null) toIdx = fromIdx;
  if ( !("frames" in settings) ) settings.frames = [];

  var i;
  for (i=fromIdx; i<=toIdx; i++) {
    if (i>=settings.frames.length) settings.frames[i] = [];
    settings.frames[i].push(path);
  }
}

var FrameBuilder = new Object();
FrameBuilder.frames = [];
FrameBuilder.layerStack = [];
FrameBuilder.prefix = "";
FrameBuilder.start = function (pathPrefix="") {
  FrameBuilder.prefix = pathPrefix;
  FrameBuilder.layerStack = [];
  FrameBuilder.frames = [];
  return FrameBuilder;
}
FrameBuilder.end = function () {
  return FrameBuilder.frames;
}
FrameBuilder.pushLayers = function (paths) {
  var i;
  for (i=0;i<paths.length;i++) {
    FrameBuilder.layerStack.push(FrameBuilder.prefix+paths[i]);
  }

  var idx = FrameBuilder.frames.length;
  FrameBuilder.frames[idx] = [];

  for (i=0;i<FrameBuilder.layerStack.length;i++) {
    FrameBuilder.frames[idx].push(FrameBuilder.layerStack[i]);
  }

  return FrameBuilder;
}
FrameBuilder.popLayers = function (count=1) {
  var i;
  for (i=0;i<count;i++) {
    FrameBuilder.layerStack.pop();
  }
  return FrameBuilder;
}
FrameBuilder.popAllLayers = function () {
  FrameBuilder.layerStack = [];
  return FrameBuilder;
}

/*
function pushLayers(paths) {
  var i;
  for (i=0;i<paths.length;i++) {
    framesBuilder.push(paths[i]);
  }

  if ( !("frames" in settings) ) settings.frames = [];
  var idx = settings.frames.length;
  settings.frames[idx] = [];

  for (i=0;i<framesBuilder.length;i++) {
    settings.frames[idx].push(framesBuilder[i]);
  }
}
function popLayers(count=1) {
  var i;
  for (i=0;i<count;i++) {
    framesBuilder.pop();
  }
}
*/

function setup(canvasContext) {
  context.canvas = canvasContext;

  context.index = 0;
  context.cacheCount = 0;
  
  context.autoplay = false;
  if ("autoplay" in settings)
    context.autoplay = settings.autoplay;

  context.timeout = 1000;
  if ("fps" in settings)
    context.timeout = 1000/settings.fps;

  if ("background" in settings && settings.background!=null && settings.background!="") {
    context.bgImg = loadImg(settings.background);
  }
  if ("play" in settings && settings.play!=null && settings.play!="") {
    context.playImg = loadImg(settings.play);
  }
  
  context.frames = settings.frames;

  update();
  refreshEvent();
}