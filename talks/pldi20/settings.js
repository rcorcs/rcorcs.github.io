settings = {
"fps" : 1,
"autoplay" : false,
"play":"frames/play.svg",

"background":"frames/bg.svg",

"frames": FrameBuilder.start("frames/")
.pushLayers(["intro/title-1.svg"])
  .pushLayers(["intro/clock.svg"])
    .pushLayers(["intro/energy.svg"])
      .pushLayers(["intro/scale.svg"])
        .popLayers()
      .pushLayers(["fade.svg","intro/scale.svg"])
.popAllLayers()
.pushLayers(["figs/intro-FM.svg"])
.popLayers()
.pushLayers(["figs/intro-SOA.svg"])
.popLayers()
.pushLayers(["figs/intro-Goal.svg"])
.popLayers()
.pushLayers(["figs/motivation-Size.svg"])
.popLayers()
.pushLayers(["figs/motivation-Size-2.svg"])
.popLayers()
.pushLayers(["figs/motivation-CodeGen.svg"])
.popLayers()
.pushLayers(["figs/eval-Compilation.svg"])
.popLayers()
.pushLayers(["figs/eval-Size.svg"])
.end()
};
