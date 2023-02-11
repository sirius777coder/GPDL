# ESM-Inpainting
----
1. Using ESMFold to generate scaffold proteins for the functional sites (which is also called inpainting). The basic formuler is $s,t \sim f_{\theta}(\hat{s},\hat{t})$ , $s,\hat{s},t,\hat{t}$ is the whole sequene, whole structures, motif seqs and motif structures. $f_{\theta}$ is the ESM-Inpainting network and its parameters.

2. Only structure module and some linear project layers(like distance linear layer and sequence output embedding) were trained.

[model](./img/inpaint.png)
-----
Author : Bo Zhang  
E-mail : zhangbo777@sjtu.edu.cn
