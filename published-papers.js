var conferences = {
"PLDI":"Conference on Programming Language Design and Implementation",
"CC":"Conference on Compiler Construction",
"CGO":"Symposium on Code Generation and Optimization",
"PACT":"Conference on Parallel Architectures and Compilation Techniques",
"HPCS":"Conference on High Performance Computing & Simulation",
"ICCS":"Conference on Computational Science",
"SBAC-PADW":"Workshops on Computer Architecture and High Performance Computing",
"SBLP":"Brazilian Symposium on Programming Languages",
"SBPO":"Brazilian Symposium of Operational Research",
};

var journals = {
"SCP":"Journal of Science of Computer Programming",
"CCPE":"Concurrency and Computation: Practice and Experience",
"JRTIP":"Journal of Real-Time Image Processing"
};


var papers = [
{
"title":"Effective function merging in the SSA form",
"authors":"Rodrigo Rocha, Pavlos Petoumenos, Zheng Wang, Murray Cole, Hugh Leather",
"conference":"PLDI",
"year":2020,
"award":null,
"keywords":["compiler","optimization","code size","function merging","IPO","link time","LTO"],
"citations":0,
"links": [
  { "name":"pdf", "url":"papers/pldi20salssa.pdf"},
  { "name":"bibtex", "url":"papers/pldi20salssa.bib"},
  { "name":"doi", "url":"https://doi.org/10.1145/3385412.3386030"},
],
},
{
"title":"Vectorization-aware loop unrolling with seed forwarding",
"authors":"Rodrigo Rocha, Vasileios Porpodas, Pavlos Petoumenos, Luis Goes, Zheng Wang, Murray Cole, Hugh Leather",
"conference":"CC",
"year":2020,
"award":null,
"keywords":["compiler","optimization","SIMD","vectorization","SLP","loop","unrolling"],
"citations":0,
"links": [
  { "name":"pdf", "url":"papers/cc20valu.pdf"},
  { "name":"bibtex", "url":"papers/cc20valu.bib"},
  { "name":"doi", "url":"https://doi.org/10.1145/3377555.3377890"},
],
},
{
"title":"Function merging by sequence alignment",
"authors":"Rodrigo Rocha, Pavlos Petoumenos, Zheng Wang, Murray Cole, Hugh Leather",
"conference":"CGO",
"year":2019,
"notes": [
  { "name":"Best Paper", "url":"docs/CGO19BestPaper.jpeg"}
],
"links": [
  { "name":"pdf", "url":"papers/cgo19fmsa.pdf"},
  { "name":"bibtex", "url":"papers/cgo19fmsa.bib"},
  { "name":"doi", "url":"https://doi.org/10.1109/CGO.2019.8661174"},
  { "name":"video", "url":"https://www.youtube.com/watch?v=sOCFYfF3iwE", "img":"img/youtube.png"}
],
"keywords":["compiler","optimization","code size","function merging","IPO","link time","LTO"],
"citations":1,
"highlight":1,
},
{
"title":"Super-Node SLP: Optimized vectorization for code sequences containing operators and their inverse elements",
"authors":"Vasileios Porpodas, Rodrigo Rocha, Evgueni Brevnov, Luís Góes, Timothy Mattson",
"conference":"CGO",
"year":2019,
"links": [
  { "name":"pdf", "url":"papers/cgo19snslp.pdf"},
  { "name":"bibtex", "url":"papers/cgo19snslp.bib"},
  { "name":"doi", "url":"https://doi.org/10.1109/CGO.2019.8661192"}
],
"keywords":["compiler","optimization","SIMD","vectorization","SLP"],
"citations":1,
},
{
"title":"VW-SLP: Auto-vectorization with adaptive vector width",
"authors":"Vasileios Porpodas, Rodrigo Rocha, Luís Góes",
"conference":"PACT",
"year":2018,
"links": [
  { "name":"pdf", "url":"papers/pact18vwslp.pdf"},
  { "name":"bibtex", "url":"papers/pact18vwslp.bib"},
  { "name":"doi", "url":"https://doi.org/10.1145/3243176.3243189"},
  { "name":"video", "url":"https://www.youtube.com/watch?v=d5LgjCDkkB0", "img":"img/youtube.png"}
],
"keywords":["compiler","optimization","SIMD","vectorization","SLP"],
"citations":4,
"highlight":2,
},
{
"title":"Look-Ahead SLP: Auto-vectorization in the presence of commutative operations",
"authors":"Vasileios Porpodas, Rodrigo Rocha, Luís Góes",
"conference":"CGO",
"year":2018,
"links": [
  { "name":"pdf", "url":"papers/cgo18lslp.pdf"},
  { "name":"bibtex", "url":"papers/cgo18lslp.bib"},
  { "name":"doi", "url":"https://doi.org/10.1145/3168807"},
  { "name":"video", "url":"https://www.youtube.com/watch?v=gIEn34LvyNo", "img":"img/youtube.png"}
],
"keywords":["compiler","optimization","SIMD","vectorization","SLP"],
"citations":5,
},
{
"title":"Automatic parallelization of recursive functions with rewriting rules",
"authors":"Rodrigo Rocha, Luís Góes, Fernando Pereira",
"journal":"SCP",
"year":2018,
"links": [
  { "name":"pdf", "url":"http://homepages.dcc.ufmg.br/~fernando/publications/papers/Rocha18_SCP.pdf"},
  { "name":"bibtex", "url":"papers/scp18apref.bib"},
  { "name":"doi", "url":"https://doi.org/10.1016/j.scico.2018.01.004"},
],
"keywords":["compiler","optimization","functional","parallelism","rewriting"],
"citations":1,
"highlight":3,
},
{
"title":"TOAST: Automatic tiling for iterative stencil computations on GPUs",
"authors":"Rodrigo Rocha, Alyson Pereira, Luiz Ramos, Luís Góes",
"journal":"CCPE",
"year":2017,
"links": [
  { "name":"pdf", "url":"papers/ccpe2017toast.pdf"},
  { "name":"bibtex", "url":"papers/ccpe2017toast.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.1002/cpe.4053"},
],
"keywords":["parallelism","tiling","optimization","stencil","skeletons","GPU"],
"citations":9,
},
{
"title":"Real-time video denoising on multicores and GPUs with Kalman-based and bilateral filters fusion",
"authors":"Sergio Pfleger, Patricia Plentz, Rodrigo Rocha, Alyson Pereira, Márcio Castro",
"journal":"JRTIP",
"year":2017,
"links": [
  { "name":"bibtex", "url":"papers/jrtip2017stmkf.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.1007/s11554-016-0659-y"},
  { "name":"video", "url":"https://www.youtube.com/watch?v=oID9Rbf6OBw", "img":"img/youtube.png"},
],
"keywords":["image processing","video processing","denoising","GPU","multicore"],
"citations":5,
},
{
"title":"Automatic partitioning of stencil computations on heterogeneous systems",
"authors":"Alyson Pereira, Rodrigo Rocha, Márcio Castro, Luís Góes, Mário Dantas",
"conference":"SBAC-PADW",
"year":2017,
"links": [
  { "name":"pdf", "url":"papers/wamca2017awp.pdf"},
  { "name":"bibtex", "url":"papers/wamca2017awp.bib"},
  { "name":"doi", "url":"https://doi.org/10.1109/SBAC-PADW.2017.16"},
],
"keywords":["parallelism","stencil","skeletons","GPU","multicore"],
"citations":2,
},
{
"title":"Extending OpenACC for efficient stencil code generation and execution by skeleton frameworks",
"authors":"Alyson Pereira, Rodrigo Rocha, Márcio Castro, Luís Góes, Mário Dantas",
"conference":"HPCS",
"year":2017,
"links": [
  { "name":"pdf", "url":"papers/hpcs2017pskelcc.pdf"},
  { "name":"bibtex", "url":"papers/hpcs2017pskelcc.bib"},
  { "name":"doi", "url":"https://doi.org/10.1109/HPCS.2017.110"},
],
"keywords":["compiler","parallelism","stencil","skeletons","GPU","multicore","OpenACC"],
"citations":7,
},
{
"title":"Enabling efficient stencil code generation in OpenACC",
"authors":"Alyson Pereira, Rodrigo Rocha, Márcio Castro, Luís Góes, Mário Dantas",
"conference":"ICCS",
"year":2017,
"links": [
  { "name":"pdf", "url":"papers/iccs2017pskelcc.pdf"},
  { "name":"bibtex", "url":"papers/iccs2017pskelcc.bib"},
  { "name":"doi", "url":"http://doi.org/10.1016/j.procs.2017.05.155"},
],
"keywords":["compiler","parallelism","stencil","skeletons","GPU","multicore","OpenACC"],
"citations":2,
},
{
"title":"Watershed-ng: an extensible distributed stream processing framework",
"authors":"Rodrigo Rocha, Bruno Hott, Vinícius Dias, Renato Ferreira, Wagner Meira, Dorgival Guedes",
"journal":"CCPE",
"year":2016,
"links": [
  { "name":"bibtex", "url":"papers/ccpe2016watershedng.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.1002/cpe.3779"},
],
"keywords":["distributed","parallelism","big data","stream processing","data flow"],
"citations":4,
},
{
"title":"An algebraic framework for parallelizing recurrence in functional programming",
"authors":"Rodrigo Rocha, Luís Góes, Fernando Pereira",
"conference":"SBLP",
"year":2016,
"links": [
  { "name":"pdf", "url":"papers/sblp2016recurrence.pdf"},
  { "name":"bibtex", "url":"papers/sblp2016recurrence.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.1007/978-3-319-45279-1_10"},
],
"keywords":["compiler","optimization","functional","parallelism","rewriting"],
"citations":1,
},
{
"title":"Distributed cycle detection in large-scale sparse graphs",
"authors":"Rodrigo Rocha, Bhalchandra Thatte",
"conference":"SBPO",
"year":2015,
"links": [
  { "name":"pdf", "url":"papers/sbpo2015cycles.pdf"},
  { "name":"bibtex", "url":"papers/sbpo2015cycles.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.13140/RG.2.1.1233.8640"},
],
"keywords":["graph theory","graphs","distributed","cycle detection"],
"citations":5,
},
{
"title":"Watershed reengineering: making streams programmable",
"authors":"Rodrigo Rocha, Renato Ferreira, Wagner Meira, Dorgival Guedes",
"conference":"SBAC-PADW",
"year":2014,
"links": [
  { "name":"pdf", "url":"papers/wpba2014watershed.pdf"},
  { "name":"bibtex", "url":"papers/wpba2014watershed.bib"},
  { "name":"doi", "url":"http://dx.doi.org/10.1109/SBAC-PADW.2014.31"},
],
"keywords":["distributed","parallelism","big data","stream processing","data flow"],
"citations":1,
},
{
"title":"Typed graph theory: Extending graphs with type systems",
"authors":"Rodrigo Rocha",
"links": [
  { "name":"pdf", "url":"papers/typed-graph-theory.pdf"},
],
"year":0,
"keywords":["graph theory","type theory","type system","typed graph theory"],
"citations":1,
"highlight":4,
},
];
