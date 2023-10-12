## InternImage: Exploring Large-Scale Vision Foundation Models with

## Deformable Convolutions

## The Chinese University of Hong Kong

###### https://github.com/OpenGVLab/InternImage

## Abstract

##### Compared to the great progress of large-scale vision

##### transformers (ViTs) in recent years, large-scale models

##### based on convolutional neural networks (CNNs) are still

##### in an early state. This work presents a new large-scale

##### CNN-based foundation model, termed InternImage, which

##### can obtain the gain from increasing parameters and train-

##### ing data like ViTs. Different from the recent CNNs that focus

##### on large dense kernels, InternImage takes deformable con-

##### volution as the core operator, so that our model not only

##### has the large effective receptive field required for down-

##### stream tasks such as detection and segmentation, but also

##### has the adaptive spatial aggregation conditioned by input

##### and task information. As a result, the proposed InternIm-

##### age reduces the strict inductive bias of traditional CNNs

##### and makes it possible to learn stronger and more robust

##### patterns with large-scale parameters from massive data like

##### ViTs. The effectiveness of our model is proven on challeng-

##### ing benchmarks including ImageNet, COCO, and ADE20K.

##### It is worth mentioning that InternImage-H achieved a new

##### record 65.4 mAP on COCO test-dev and 62.9 mIoU on

##### ADE20K, outperforming current leading CNNs and ViTs.

## 1. Introduction

##### With the remarkable success of transformers in large-

##### scale language models [3–8], vision transformers (ViTs) [2,

##### 9–15] have also swept the computer vision field and are

##### becoming the primary choice for the research and prac-

##### tice of large-scale vision foundation models. Some pio-

##### neers [16–20] have made attempts to extend ViTs to very

##### large models with over a billion parameters, beating convo-

##### lutional neural networks (CNNs) and significantly pushing

##### the performance bound for a wide range of computer vision

```
* equal contribution,Bcorresponding author (qiaoyu@pjlab.org.cn)
```
```
(b) localattention
✗long-range dependence
✓adaptive spatialaggregation
✓computation/memory efficient
```
```
(d)dynamic sparse kernel (ours)
✓long-range dependence
✓adaptive spatialaggregation
✓computation/memory efficient
```
```
(a) globalattention
✓long-range dependence
✓adaptive spatialaggregation
✗computation/memory efficient
```
```
(c) large kernel
✓long-range dependence
✗adaptive spatialaggregation
✓computation/memory efficient
```
```
querypixels responsepixelswith fixed weights
responsepixelswith adaptive weights
```
###### Figure 1.Comparisons of different core operators.(a) shows

###### the global aggregation of multi-head self-attention (MHSA) [1],

###### whose computational and memory costs are expensive in down-

###### stream tasks that require high-resolution inputs. (b) limits the

###### range of MHSA into a local window [2] to reduce the cost. (c)

###### is a depth-wise convolution with very large kernels to model long-

###### range dependencies. (d) is a deformable convolution, which shares

###### similar favorable properties with MHSA and is efficient enough

###### for large-scale models. We start from it to build a large-scale CNN.

##### tasks, including basic classification, detection, and segmen-

##### tation. While these results suggest that CNNs are inferior

##### to ViTs in the era of massive parameters and data, we ar-

##### gue thatCNN-based foundation models can also achieve

##### comparable or even better performance than ViTs when

# arXiv:2211.05778v4 [cs.CV] 17 Apr 2023


##### equipped with similar operator-/architecture-level designs,

##### scaling-up parameters, and massive data.

##### To bridge the gap between CNNs and ViTs, we first

##### summarize their differences from two aspects: (1) From

##### the operator level [9, 21, 22], the multi-head self-attention

##### (MHSA) of ViTs has long-range dependencies and adap-

##### tive spatial aggregation (see Fig. 1(a)). Benefiting from the

##### flexible MHSA, ViTs can learn more powerful and robust

##### representations than CNNs from massive data. (2) From

##### the architecture view [9, 22, 23], besides MHSA, ViTs con-

##### tain a series of advanced components that are not included

##### in standard CNNs, such as Layer Normalization (LN) [24],

##### feed-forward network (FFN) [1], GELU [25], etc. Although

##### recent works [21, 22] have made meaningful attempts to in-

##### troduce long-range dependencies into CNNs by using dense

##### convolutions with very large kernels (e.g., 31×31) as shown

##### in Fig. 1 (c), there is still a considerable gap with the state-

##### of-the-art large-scale ViTs [16, 18–20, 26] in terms of per-

##### formance and model scale.

##### In this work, we concentrate on designing a CNN-based

##### foundation model that can efficiently extend to large-scale

##### parameters and data. Specifically, we start with a flexible

##### convolution variant—deformable convolution (DCN) [27,

##### 28]. By combining it with a series of tailored block-

##### level and architecture-level designs similar to transformers,

##### we design a brand-new convolutional backbone network,

##### termedInternImage. As shown in Fig. 1, different from

##### recently improved CNNs with very large kernels such as

##### 31 ×31 [22], the core operator of InternImage is a dynamic

##### sparse convolution with a common window size of 3×3, (1)

##### whose sampling offsets are flexible to dynamically learn ap-

##### propriate receptive fields (can be long- or short-range) from

##### given data; (2) the sampling offsets and modulation scalars

##### are adaptively adjusted according to the input data, which

##### can achieve adaptive spatial aggregation like ViTs, reduc-

##### ing the over-inductive bias of regular convolutions; and (3)

##### the convolution window is a common 3×3, avoiding the

##### optimization problems and expensive costs caused by large

##### dense kernels [22, 29].

##### With the aforementioned designs, the proposed Intern-

##### Image can efficiently scale to large parameter sizes and

##### learn stronger representations from large-scale training

##### data, achieving comparable or even better performance to

##### large-scale ViTs [2, 11, 30] on a wide range of vision tasks.

##### In summary, our main contributions are as follows:

##### (1) We present a new large-scale CNN-based founda-

##### tion model—InternImage. To our best knowledge, it is the

##### first CNN that effectively scales to over 1 billion parameters

##### and 400 million training images and achieves comparable or

##### even better performance than state-of-the-art ViTs, showing

##### that convolutional models are also a worth-exploring direc-

##### tion for large-scale model research.

##### (2) We successfully scale CNNs to large-scale settings

```
65.
```
```
63.
63.7 64.
62.
```
###### 45

###### 47

###### 49

###### 51

###### 53

###### 55

###### 57

###### 59

###### 61

###### 63

###### 65

###### 67

###### 0 0.5 1 1.5 2 2.5 3 3.

###### COCO box AP (%)

###### #parameter (B)

```
InternImage-H (test-dev)
SwinV2 (test-dev)
FD-SwinV2-G (test-dev)
BEiT-3 (test-dev)
Florence-CoSwin-H (test-dev)
InternImage (val2017)
Swin (val2017)
ConvNeXt (val2017)
```
###### Figure 2. Performance comparison on COCO of different

###### backbones.The proposed InternImage-H achieves a new record

###### 65.4 box AP on COCO test-dev, significantly outperforming state-

###### of-the-art CNNs and large-scale ViTs.

##### by introducing long-range dependencies and adaptive spa-

##### tial aggregation using an improved 3×3 DCN operator, and

##### explore the tailored basic block, stacking rules, and scaling

##### strategies centered on the operator. These designs make ef-

##### fective use of the operator, enabling our models to obtain

##### the gains from large-scale parameters and data.

##### (3) We evaluate the proposed model on representative

##### vision tasks including image classification, object detec-

##### tion, instance and semantic segmentation, and compared it

##### with state-of-the-art CNNs and large-scale ViTs by scal-

##### ing the model size ranging from 30 million to 1 billion,

##### the data ranging from 1 million to 400 million. Specifi-

##### cally, our model with different parameter sizes can consis-

##### tently outperform prior arts on ImageNet [31]. InternImage-

##### B achieves 84.9% top-1 accuracy trained only on the

##### ImageNet-1K dataset, outperforming CNN-based counter-

##### parts [21, 22] by at least 1.1 points. With large-scale pa-

##### rameters (i.e., 1 billion) and training data (i.e., 427 million),

##### the top-1 accuracy of InternImage-H is further boosted to

##### 89.6%, which is close to well-engineering ViTs [2, 30] and

##### hybrid-ViTs [20]. In addition, on COCO [32], a challeng-

##### ing downstream benchmark, our best model InternImage-H

##### achieves state-of-the-art 65.4% box mAP with 2.18 billion

##### parameters, 2.3 points higher than SwinV2-G [16] (65.4vs.

##### 63.1) with 27% fewer parameters as shown in Fig. 2.

### 2. Related Work

##### Vision foundation models. Convolutional neural net-

##### works (CNNs) became the mainstream for visual recogni-

##### tion after the large-scale dataset and computation resources

##### were available. Straining from AlexNet [33], lots of deeper


##### and more effective neural network architectures have been

##### proposed, such as VGG [34], GoogleNet [35], ResNet [36],

##### ResNeXt [37], EfficientNet [38, 39], etc. In addition to the

##### architectural design, more sophisticated convolution opera-

##### tions such as depth-wise convolution [40] and deformable

##### convolution [27, 28] are formulated. By considering the

##### advanced designs of transformers, modern CNNs showed

##### promising performance on the vision tasks by discover-

##### ing better components in macro/micro designs and intro-

##### ducing improved convolutions with long-range dependen-

##### cies [21, 41–43] or dynamic weights [44].

##### In recent years, a new line of vision foundation mod-

##### els focuses on transformer-based architecture. ViT [9] is

##### the most representative model, which achieves great suc-

##### cess in vision tasks thanks to global receptive fields and

##### dynamic spatial aggregation. However, global attention in

##### ViT suffers from expensive computational/memory com-

##### plexity, especially on large feature maps, which limits its

##### application in downstream tasks. To address this problem,

##### PVT [10, 11] and Linformer [45] performed global atten-

##### tion on the downsampled key and value maps, DAT [46]

##### employed deformable attention to sparsely sample informa-

##### tion from value maps, while HaloNet [47] and Swin trans-

##### former [2] developed local attention mechanisms and used

##### haloing and shift operations to transfer information among

##### adjacent local regions.

##### Large-scale models.Scaling up models is an important

##### strategy to improve feature representation quality, which

##### has been well-studied in the natural language processing

##### (NLP) domain [48]. Inspired by the success in the NLP

##### field, Zhaiet al. [19] first extended ViT to 2 billion pa-

##### rameters. Liuet al.[16] enlarged the hierarchical-structure

##### Swin transformer to a deeper and wider model with 3 bil-

##### lion parameters. Some researchers developed large-scale

##### hybrid ViTs [20, 49] by combining the advantages of ViTs

##### and CNNs at different levels. Recently, BEiT-3 [17] further

##### explored stronger representations based on ViT with large-

##### scale parameters using multimodal pre-training. These

##### methods significantly raise the upper bound of basic vision

##### tasks. However, research on CNN-based large-scale models

##### has lagged behind transformer-based architectures in terms

##### of the total number of parameters and performance. Al-

##### though newly-proposed CNNs [21, 41–43] introduce long-

##### range dependencies by using convolutions with very large

##### kernels or recursive gated kernels, there is still a consider-

##### able gap with state-of-the-art ViTs. In this work,we aim

##### to develop a CNN-based foundation model that can extend

##### efficiently to a large scale comparable to ViT.

### 3. Proposed Method

##### To design a large-scale CNN-based foundation model,

##### we start with a flexible convolution variant, namely de-

##### formable convolution v2 (DCNv2) [28] and make some

##### tune-ups based on it to better suit the requirements of large-

##### scale foundation models. Then, we build the basic block

##### by combining the tuned convolution operator with advanced

##### block designs used in modern backbones [16, 19]. Finally,

##### we explore the stacking and scaling principles of DCN-

##### based blocks to build a large-scale convolutional model that

##### can learn strong representations from massive data.

#### 3.1. Deformable Convolution v

##### Convolutionvs. MHSA.Previous works [21, 22, 50]

##### have extensively discussed the differences between CNNs

##### and ViTs. Before deciding on the core operator of InternIm-

##### age, we first summarize the main differences between regu-

##### lar convolution and MHSA.

##### (1)Long-range dependencies. Although it has long been

##### recognized that models with large effective receptive fields

##### (long-range dependencies) usually perform better on down-

##### stream vision tasks [51–53], the de-facto effective receptive

##### field of CNNs [34, 36] stacked by 3×3 regular convolutions

##### is relatively small. Even with very deep models, the CNN-

##### based model still cannot acquire long-range dependencies

##### like ViTs, which limits its performance.

##### (2)Adaptive spatial aggregation. Compared to MHSA

##### whose weights are dynamically conditioned by the input,

##### regular convolution [54] is an operator with static weights

##### and strong inductive biases such as 2D locality, neigh-

##### borhood structure, translation equivalence, etc. With the

##### highly-inductive properties, models composed by regular

##### convolutions might converge faster and require less train-

##### ing data than ViTs, but it also restricts CNNs from learning

##### more general and robust patterns from web-scale data.

##### Revisiting DCNv2.A straightforward way to bridge the

##### gap between convolution and MHSA is to introduce long-

##### range dependencies and adaptive spatial aggregation into

##### regular convolutions. Let us start with DCNv2 [28], which

##### is a general variant of regular convolution. Given an input

##### x∈RC×H×Wand current pixelp 0 , DCNv2 can be formu-

##### lated as:

##### y(p 0 ) =

##### ∑K

```
k=
```
##### wkmkx(p 0 +pk+ ∆pk), (1)

##### whereKrepresents the total number of sampling points,

##### andkenumerates the sampling point. wk∈RC×Cde-

##### notes the projection weights of thek-th sampling point,

##### andmk∈Rrepresents the modulation scalar of thek-

##### th sampling point, which is normalized by sigmoid func-

##### tion. pkdenotes thek-th location of the pre-defined grid

##### sampling{(− 1 ,−1),(− 1 ,0),...,(0,+1),...,(+1,+1)}as

##### in regular convolutions, and∆pkis the offset correspond-

##### ing to thek-th grid sampling location. We see from the

##### equation that (1) for long-range dependencies, the sampling

##### offset∆pkis flexible and able to interact with short- or


###### stem

###### downsampling

###### stage 𝟐

###### basic block ×𝐿!

###### downsampling

###### downsampling

###### stage 𝟏

###### basic block ×𝐿"

###### stage 𝟑

###### basic block ×𝐿#

###### stage 𝟒

###### basic block ×𝐿$

###### 𝐻×𝑊× 3

###### 𝐻/ 4 ×𝑊/ 4 ×𝐶"

###### 𝐻/ 8 ×𝑊/ 8 ×𝐶!

###### 𝐻/ 16 ×𝑊/ 16 ×𝐶#

###### 𝐻/ 32 ×𝑊/ 32 ×𝐶$

##### cls, det, seg, ...

###### stacking rules

###### (1) 𝐶%= 2 %&"𝐶"

###### (2) 𝐺%=𝐶%/𝐶′

###### (3) 𝐿"=𝐿!=𝐿$

###### (4) 𝐿"≤𝐿#

###### stem

###### 3 × 3 conv, s 2 , p 1

###### LN, GELU

###### 3 × 3 conv, s 2 , p 1

```
LN
```
###### downsampling

###### 3 × 3 conv, s 2 , p 1

###### LN

###### 𝐿%×

###### Δ𝑝,𝐦

###### LN

###### FFN

###### LN

###### DCNv3 (𝐺%)

##### stage 𝒊

###### Figure 3.Overall Architecture of InternImage, where the core

###### operator is DCNv3, and the basic block composes of layer normal-

###### ization (LN) [24] and feed-forward network (FFN) [1] as trans-

###### formers, the stem and downsampling layers follows conventional

###### CNN’s designs, where “s2” and “p1” mean stride 2 and padding

###### 1, respectively. Constrained by the stacking rules, only 4 hyper-

###### parameters(C 1 , C′, L 1 , L 3 )can decide a model variant.

##### long-range features; and (2) for adaptive spatial aggrega-

##### tion, both the sampling offset∆pkand modulation scalar

##### mkare learnable and conditioned by inputx. So it can be

##### found thatDCNv2 shares similar favorable properties with

##### MHSA, which motivated us to develop large-scale CNN-

##### based foundation models on the basis of this operator.

##### Extending DCNv2 for Vision Foundation Models.In

##### common practice, DCNv2 is usually used as an extension

##### to regular convolutions, loading pre-trained weights of reg-

##### ular convolutions and fine-tuning for better performance,

##### which is not exactly suitable for large-scale vision founda-

##### tion models that need to be trained from scratch. In this

##### work, to address this problem, we extend DCNv2 from as-

##### pects as follows:

##### (1)Sharing weights among convolutional neurons.Sim-

##### ilar to regular convolution, different convolutional neu-

##### rons^1 in original DCNv2 have independent linear projection

##### weights, and thus its parameter and memory complexity

##### are linear with the total number of sampling points, which

##### significantly limits the efficiency of the model, especially

##### in large-scale models. To remedy this problem, we bor-

##### row the idea from the separable convolution [55] and de-

##### tach the original convolution weightswkinto depth-wise

##### and point-wise parts, where the depth-wise part is respon-

##### sible by the original location-aware modulation scalarmk,

##### and the point-wise part is the shared projection weightsw

##### among sampling points.

##### (2)Introducing multi-group mechanism. The multi-

##### group (head) design first appeared in group convolu-

##### tion [33], and it is widely used in MHSA [1] of transformers

##### and works with adaptive spatial aggregation to effectively

##### learn richer information from different representation sub-

##### spaces at different locations. Inspired by this, we split the

##### spatial aggregation process intoGgroups, each of which

##### has individual sampling offsets∆pgkand modulation scale

##### mgk, and thus different groups on a single convolution layer

##### can have different spatial aggregation patterns, resulting in

##### stronger features for downstream tasks.

##### (3)Normalizing modulation scalars along sampling

##### points.The modulation scalars in the original DCNv2 are

##### element-wise normalized by the sigmoid function. There-

##### fore, each modulation scalar is in the range [0, 1], and the

##### sum of the modulation scalars of all sample points is not sta-

##### ble and varies from 0 toK. This leads to unstable gradients

##### in DCNv2 layers when training with large-scale parame-

##### ters and data. To alleviate the instability issues, we change

##### element-wise sigmoid normalization to softmax normaliza-

##### tion along sample points. In this way, the sum of the modu-

##### lation scalars is constrained to 1, which makes the training

##### process of models at different scales more stable.

##### Combining the aforementioned modifications, the ex-

##### tended DCNv2, marked as DCNv3, can be formulated as

##### Eqn. (2).

##### y(p 0 ) =

##### ∑G

```
g=
```
##### ∑K

```
k=
```
##### wgmgkxg(p 0 +pk+ ∆pgk), (2)

##### whereGdenotes the total number of aggregation groups.

##### For theg-th group,wg ∈RC×C

```
′
```
##### denotes the location-

##### irrelevant projection weights of the group, whereC′=C/G

##### represents the group dimension.mgk∈Rdenotes the mod-

##### ulation scalar of thek-th sampling point in theg-th group,

##### normalized by the softmax function along the dimensionK.

##### xg∈RC

```
′×H×W
```
##### represents the sliced input feature map.

##### ∆pgkis the offset corresponding to the grid sampling loca-

##### tionpkin theg-th group.

##### In general, DCNv3, as an extension of the DCN series,

##### enjoys three merits as follows: (1) This operator made up

(^1) A 3×3 regular convolution has 9 linear projection neurons.


##### for the deficiencies of regular convolution in terms of long-

##### range dependencies and adaptive spatial aggregation; (2)

##### Compared with attention-based operators such as common

##### MHSA and closely-related deformable attention [46, 56],

##### this operator inherits the inductive bias of convolution,

##### making our model more efficient with fewer training data

##### and shorter training time; (3) This operator is based on

##### sparse sampling, which is more computational and mem-

##### ory efficient than previous methods such as MHSA [1] and

##### re-parameterizing large kernel [22]. In addition, due to

##### the sparse sampling, DCNv3 only needs a 3×3 kernel to

##### learn long-range dependencies, which is easier to be op-

##### timized and avoids extra auxiliary techniques such as re-

##### parameterizing [22] used in large kernels.

#### 3.2. InternImage Model

##### Using DCNv3 as the core operator brings a new prob-

##### lem:how to build a model that can make effective use of the

##### core operator? In this section, we first present the details

##### of the basic block and other integral layers of our model,

##### and then we construct a new CNN-based foundation model

##### termed InternImage, by exploring a tailored stacking strat-

##### egy for these basic blocks. Finally, we study scaling-up

##### rules for the proposed model to obtain the gain from in-

##### creasing parameters.

##### Basic block.Unlike the widely used bottlenecks in tradi-

##### tional CNNs [36], the design of our basic block is closer to

##### ViTs, which is equipped with more advanced components

##### including LN [24], feed-forward networks (FFN) [1], and

##### GELU [25]. This design is proved to be efficient [2, 10,

##### 11, 21, 22] in various vision tasks. The details of our ba-

##### sic block are illustrated in Fig. 3, where the core operator

##### is DCNv3, and the sampling offsets and modulation scales

##### are predicted by passing input featurexthrough a separable

##### convolution (a 3×3 depth-wise convolution followed by a

##### linear projection). For other components, we use the post-

##### normalization setting [57] by default and follow the same

##### design as that of the plain transformer [1, 9].

##### Stem & downsampling layers.To obtain hierarchical

##### feature maps, we use convolutional stem and downsampling

##### layers to resize the feature maps to different scales. As

##### shown in Fig. 3, the stem layer is placed before the first

##### stage to reduce the input resolution by 4 times. It consists

##### of two convolutions, two LN layers, and one GELU layer,

##### where the kernel size of the two convolutions is 3, the stride

##### is 2, the padding is 1, and the output channel of the first con-

##### volution is half of the second one. Similarly, the downsam-

##### pling layer is made up of a 3×3 convolution with a stride

##### of 2 and a padding of 1, followed by one LN layer. It sits

##### between the two stages and is used to downsample the input

##### feature map by 2 times.

##### Stacking rules. To clarify the block-stacking process,

##### we first list the integral hyperparameters of the InternImage

```
model name C 1 C′ L 1 , 2 , 3 , 4 #params
InternImage-T (origin) 64 16 4, 4, 18, 4 30M
InternImage-S 80 16 4, 4, 21, 4 50M
InternImage-B 112 16 4, 4, 21, 4 97M
InternImage-L 160 16 5, 5, 22, 5 223M
InternImage-XL 192 16 5, 5, 24, 5 335M
InternImage-H 320 32 6, 6, 32, 6 1.08B
```
###### Table 1. Hyper-parameters for models of different scales.

###### InternImage-T is the origin model, and -S/B/L/XL/H are scaled

###### up from -T. “#params” denotes the number of parameters.

##### as follows:

##### Ci: the channel number of thei-th stage;

##### Gi: the group number of the DCNv3 in thei-th stage;

##### Li: the number of basic blocks in thei-th stage.

##### Since our model has 4 stages, a variant is decided by 12

##### hyper-parameters, whose search space is too large to ex-

##### haustively enumerate and find the best variant. To reduce

##### the search space, we summarize the design experiences of

##### prior arts [2, 21, 36] into 4 rules as shown in Fig. 3, where

##### the first rule makes the channel numbers of the last three

##### stages determined by the channel numberC 1 of the first

##### stage, and the second rule lets the group number correspond

##### to the channel number of stages. For the number of stacked

##### blocks in different stages, we simplify the stacking pattern

##### to “AABA”, which means the block number of stage 1, 2,

##### and 4 are the same, and are not greater than that of the stage

##### 3 as illustrated in the last two rules. With these rules, a

##### InternImage variant can be defined by using only 4 hyper-

##### parameters(C 1 ,C′,L 1 ,L 3 ).

##### Let us choose a model with 30 million parameters as the

##### origin and discretizeC 1 to{ 48 , 64 , 80 },L 1 to{ 1 , 2 , 3 , 4 , 5 },

##### andC′to{ 16 , 32 }. In this way, the original huge search

##### space is reduced to 30, and we can find the best model

##### from the 30 variants by training and evaluating them in Im-

##### ageNet [31]. In practice, we use the best hyper-parameter

##### setting(64, 16 , 4 ,18)to define the origin model and scale it

##### to different scales.

##### Scaling rules. Based on the optimal origin model un-

##### der the aforementioned constraints, we further explore the

##### parameter scaling rules inspired by [38]. Specifically, we

##### consider two scaling dimensions: depthD(i.e., 3 L 1 +L 3 )

##### and widthC 1 , and scale the two dimensions usingα,βand

##### a composite factorφ. The scaling rules can be written as:

##### D′=αφDandC 1 ′ =βφC 1 , whereα≥ 1 ,β≥ 1 , and

##### αβ^1.^99 ≈ 2. Here, 1.99 is specific for InternImage and cal-

##### culated by doubling the model width and keeping the depth

##### constant. We experimentally find out that the best scaling

##### setting isα= 1. 09 andβ= 1. 36 , and then we base on it

##### to construct InternImage variants with different parameter

##### scales, namely InternImage-T/S/B/L/XL, whose complex-

##### ity is similar to those of ConvNeXt [21]. To further test the

##### capability, we built a larger InternImage-H with 1 billion


##### parameters, and to accommodate very large model widths,

##### we also change the group dimensionC′to 32. The config-

##### urations are summarized in Table 1.

### 4. Experiment

##### We analyze and compare InternImage with the leading

##### CNNs and ViTs on representative vision tasks including im-

##### age classification, object detection, instance and semantic

##### segmentation. Besides the experiments in the main paper,

##### due to space constraints, more experimental setups and ab-

##### lation studies are presented in the supplementary material.

#### 4.1. Image Classification

##### Settings.We evaluate the classification performance of

##### InternImage on ImageNet [31]. For fair comparisons, fol-

##### lowing common practices [2, 10, 21, 58], InternImage-T/S/B

##### are trained on ImageNet-1K (∼1.3 million) for 300 epochs,

##### and InternImage-L/XL are first trained on ImageNet-22K

##### (∼14.2 million) for 90 epochs and then fine-tuned on

##### ImageNet-1K for 20 epochs. To further explore the ca-

##### pability of our model and match the large-scale private

##### data used in previous methods [16, 20, 59], we adopt M3I

##### Pre-training [60], a unified pre-training approach available

##### for both unlabeled and weakly-labeled data, to pre-train

##### InternImage-H on a 427 million joint dataset of public

##### Laion-400M [61], YFCC-15M [62], and CC12M [63] for

##### 30 epochs, and then we fine-tune the model on ImageNet-

##### 1K for 20 epochs.

##### Results.Table 2 shows the classification results of mod-

##### els with different scales. With similar parameters and com-

##### putational costs, our models are comparable or even su-

##### perior to the state-of-the-art transformer-based and CNN-

##### based models. For example, InternImage-T achieves 83.5%

##### top-1 accuracy, outperforming ConvNext-T [21] with a

##### clear margin of 1.4 points. InternImage-S/B keeps the

##### leading position and InternImage-B surpasses the hybrid-

##### ViT CoAtNet-2 [20] by 0.8 points. When pre-trained on

##### ImageNet-22K and the large-scale joint dataset, the top-

##### accuracy of InternImage-XL and -H are boosted to 88.0%

##### and 89.6%, respectively, which is better than previous

##### CNNs [22, 67] also trained with large-scale data, and closes

##### the gap with the state-of-the-art large-scale ViTs to about 1

##### point. This gap may be caused by the discrepancy between

##### large-scale inaccessible private data and the aforementioned

##### joint public data. These results show that our InternImage

##### not only has good performance on the common parameter

##### scale and the public training data, but also can effectively

##### extend to large-scale parameters and data.

#### 4.2. Object Detection

##### Settings. We verify the detection performance of our

##### InternImage on the COCO benchmark [32], on top of

```
method type scale#params #FLOPsacc (%)
DeiT-S [58] T 2242 22M 5G 79.
PVT-S [10] T 2242 25M 4G 79.
Swin-T [2] T 2242 29M 5G 81.
CoAtNet-0 [20] T 2242 25M 4G 81.
CSwin-T [12] T 2242 23M 4G 82.
PVTv2-B2 [11] T 2242 25M 4G 82.
DeiT III-S [64] T 2242 22M 5G 81.
SwinV2-T/8 [16] T 2562 28M 6G 81.
Focal-T [65] T 2242 29M 5G 82.
ConvNeXt-T [21] C 2242 29M 5G 82.
ConvNeXt-T-dcls [66] C 2242 29M 5G 82.
SLaK-T [29] C 2242 30M 5G 82.
HorNet-T [43] C 2242 23M 4G 83.
InternImage-T (ours) C 2242 30M 5G 83.
PVT-L [10] T 2242 61M 10G 81.
Swin-S [2] T 2242 50M 9G 83.
CoAtNet-1 [20] T 2242 42M 8G 83.
PVTv2-B4 [11] T 2242 63M 10G 83.
SwinV2-S/8 [16] T 2562 50M 12G 83.
ConvNeXt-S [21] C 2242 50M 9G 83.
ConvNeXt-S-dcls [66] C 2242 50M 10G 83.
SLaK-S [29] C 2242 55M 10G 83.
HorNet-S [43] C 2242 50M 9G 84.
InternImage-S (ours) C 2242 50M 8G 84.
DeiT-B [58] T 2242 87M 18G 83.
Swin-B [2] T 2242 88M 15G 83.
CoAtNet-2 [20] T 2242 75M 16G 84.
PVTv2-B5 [11] T 2242 82M 12G 83.
DeiT III-B [64] T 2242 87M 18G 83.
SwinV2-B/8 [16] T 2562 88M 20G 84.
RepLKNet-31B [22] C 2242 79M 15G 83.
ConvNeXt-B [21] C 2242 88M 15G 83.
ConvNeXt-B-dcls [66] C 2242 89M 17G 84.
SLaK-B [29] C 2242 95M 17G 84.
HorNet-B [43] C 2242 88M 16G 84.
InternImage-B (ours) C 2242 97M 16G 84.
Swin-L‡[2] T 3842 197M 104G 87.
CoAtNet-3‡[20] T 3842 168M 107G 87.
CoAtNet-4‡[20] T 3842 275M 190G 87.
DeiT III-L‡[64] T 3842 304M 191G 87.
SwinV2-L/24‡[16] T 3842 197M 115G 87.
RepLKNet-31L‡[22] C 3842 172M 96G 86.
HorNet-L‡[43] C 3842 202M 102G 87.
ConvNeXt-L‡[21] C 3842 198M 101G 87.
ConvNeXt-XL‡[21] C 3842 350M 179G 87.
InternImage-L‡(ours) C 3842 223M 108G 87.
InternImage-XL‡(ours) C 3842 335M 163G 88.
ViT-G/14#[30] T 5182 1.84B 5160G 90.
CoAtNet-6#[20] T 5122 1.47B 1521G 90.
CoAtNet-7#[20] T 5122 2.44B 2586G 90.
Florence-CoSwin-H#[59] T − 893M − 90.
SwinV2-G#[16] T 6402 3.00B − 90.
RepLKNet-XL#[22] C 3842 335M 129G 87.
BiT-L-ResNet152x4#[67] C 4802 928M − 87.
InternImage-H#(ours) C 2242 1.08B 188G 88.
InternImage-H#(ours) C 6402 1.08B 1478G 89.
```
###### Table 2.Image classification performance on the ImageNet val-

###### idation set. “type” refers to model type, where “T” and “C” de-

###### note transformer and CNN, respectively. “scale” is the input scale.

###### “acc” is the top-1 accuracy. “‡” indicates the model is pre-trained

###### on ImageNet-22K [31]. “#” indicates pretraining on extra large-

###### scale private dataset such as JFT-300M [68], FLD-900M [59], or

###### the joint public dataset in this work.


```
method #params #FLOPs Mask R-CNN 1×schedule Mask R-CNN 3×+MS schedule
APb APb 50 APb 75 APm APm 50 APm 75 APb APb 50 APb 75 APm APm 50 APm 75
Swin-T [2] 48M 267G 42.7 65.2 46.8 39.3 62.2 42.2 46.0 68.1 50.3 41.6 65.1 44.
ConvNeXt-T [21] 48M 262G 44.2 66.6 48.3 40.1 63.3 42.8 46.2 67.9 50.8 41.7 65.0 44.
PVTv2-B2 [11] 45M 309G 45.3 67.1 49.6 41.2 64.2 44.4 47.8 69.7 52.6 43.1 66.8 46.
ViT-Adapter-S [69] 48M 403G 44.7 65.8 48.3 39.9 62.5 42.8 48.2 69.7 52.5 42.8 66.4 45.
InternImage-T (ours) 49M 270G 47.2 69.0 52.1 42.5 66.1 45.8 49.1 70.4 54.1 43.7 67.3 47.
Swin-S [2] 69M 354G 44.8 66.6 48.9 40.9 63.4 44.2 48.2 69.8 52.8 43.2 67.0 46.
ConvNeXt-S [21] 70M 348G 45.4 67.9 50.0 41.8 65.2 45.1 47.9 70.0 52.7 42.9 66.9 46.
PVTv2-B3 [11] 65M 397G 47.0 68.1 51.7 42.5 65.7 45.7 48.4 69.8 53.3 43.2 66.9 46.
InternImage-S (ours) 69M 340G 47.8 69.8 52.8 43.3 67.1 46.7 49.7 71.1 54.5 44.5 68.5 47.
Swin-B [2] 107M 496G 46.9 −− 42.3 −− 48.6 70.0 53.4 43.3 67.1 46.
ConvNeXt-B [21] 108M 486G 47.0 69.4 51.7 42.7 66.3 46.0 48.5 70.1 53.3 43.5 67.1 46.
PVTv2-B5 [11] 102M 557G 47.4 68.6 51.9 42.5 65.7 46.0 48.4 69.2 52.9 42.9 66.6 46.
ViT-Adapter-B [69] 120M 832G 47.0 68.2 51.4 41.8 65.1 44.9 49.6 70.6 54.0 43.6 67.7 46.
InternImage-B (ours) 115M 501G 48.8 70.9 54.0 44.0 67.8 47.4 50.3 71.4 55.3 44.8 68.7 48.
```
```
method #param #FLOPs Cascade Mask R-CNN 1×schedule Cascade Mask R-CNN 3×+MS schedule
Swin-L‡[2] 253M 1382G 51.8 71.0 56.2 44.9 68.4 48.9 53.9 72.4 58.8 46.7 70.1 50.
ConvNeXt-L‡[21] 255M 1354G 53.5 72.8 58.3 46.4 70.2 50.2 54.8 73.8 59.8 47.6 71.3 51.
RepLKNet-31L‡[22] 229M 1321G −−−−−− 53.9 72.5 58.6 46.5 70.0 50.
HorNet-L‡[43] 259M 1358G −−−−−− 56.0 −− 48.6 −−
InternImage-L‡(ours) 277M 1399G 54.9 74.0 59.8 47.7 71.4 52.1 56.1 74.8 60.7 48.5 72.4 53.
ConvNeXt-XL‡[21] 407M 1898G 53.6 72.9 58.5 46.5 70.3 50.5 55.2 74.2 59.9 47.7 71.6 52.
InternImage-XL‡(ours) 387M 1782G 55.3 74.4 60.1 48.1 71.9 52.4 56.2 75.0 61.2 48.8 72.5 53.
```
###### Table 3.Object detection and instance segmentation performance on COCO val2017 .The FLOPs are measured with 1280× 800

###### inputs. APband APmrepresent box AP and mask AP, respectively. “MS” means multi-scale training.

##### two representative object detection frameworks: Mask R-

##### CNN [70], and Cascade Mask R-CNN [71]. We follow

##### common practices [2, 11] to initialize the backbone with

##### pre-trained classification weights, and train models use a

##### 1 ×(12 epochs) or 3×(36 epochs) schedule by default.

##### Results. As shown in Table 3, when using Mask R-

##### CNN for object detection, we find that under a compara-

##### ble number of parameters, our models significantly surpass

##### their counterparts. For example, with the 1 ×training sched-

##### ule, the box AP (APb) of InternImage-T is 4.5 points better

##### than Swin-T [2] (47.2vs. 42.7), and 3.0 points higher than

##### ConvNeXt-T [21] (47.2vs. 44.2). With the 3×multi-scale

##### training schedule, more parameters, and more advanced

##### Cascade Mask R-CNN [71], InternImage-XL achieves APb

##### of 56.2, surpassing ConvNeXt-XL by 1.0 points (56.2vs.

##### 55.2). Similar results are also seen in instance segmentation

##### experiments. With the 1×training schedule, InternImage-T

##### yields 42.5 mask AP (i.e., APm), which outperforms Swin-

##### T and ConvNeXt-T by 3.2 points (42.5vs. 39.3) and 2.

##### points (42.5vs. 40.1), respectively. The best APm48.8 is

##### obtained by InternImage-XL with Cascade Mask R-CNN,

##### which is at least 1.1 points higher than its counterparts.

##### To further push the performance bound of object detec-

##### tion, we follow the advanced setting used in leading meth-

##### ods [16, 17, 26, 74, 78] to initialize the backbone with the

##### weights pre-trained on ImageNet-22K or the large-scale

##### joint dataset, and double its parameters via the composite

##### techniques [78] (see the model with 2 billion parameters in

##### Fig. 2). Then, we fine-tune it along with the DINO [74]

```
method detector #params
```
```
APb
val2017test-dev
Swin-L [2] DyHead [72] 213M 56.2 58.
Swin-L‡[2] HTC++ [2] 284M 58.0 58.
Swin-L‡[2] Soft-Teacher [73] 284M 60.7 61.
Florence-CoSwin-H#[59]DyHead [72] 637M 62.0 62.
ViT-L‡[9] ViT-Adapter [69] 401M 62.6 62.
Swin-L‡[2] DINO [74] 218M 63.2 63.
FocalNet-H‡[75] DINO [74] 746M 64.2 64.
ViT-Huge [76] Group-DETRv2 [76]629M − 64.
SwinV2-G#[16] HTC++ [2] 3.00B 62.5 63.
BEiT-3#[17] ViTDet [77] 1.90B − 63.
FD-SwinV2-G#[26] HTC++ [2] 3.00B − 64.
InternImage-XL‡(ours) DINO [74] 602M 64.2 64.
InternImage-H#(ours) DINO [74] 2.18B 65.0 65.
```
###### Table 4. Comparison of the state-of-the-art detectors on

###### COCO val2017 and test-dev.

##### detector on the Objects365 [79] and COCO datasets one af-

##### ter another for 26 epochs and 12 epochs, respectively. As

##### shown in Table 4, our method achieves the best results of

##### 65.0 APband 65.4 APbon COCO val2017 and test-dev.

##### Compared to previous state-of-the-art models, we surpass

##### FD-SwinV2-G [26] by 1.2 points (65.4vs. 64.2), with 27%

##### fewer parameters and without complicated distillation pro-

##### cesses, which shows the effectiveness of our models on the

##### detection task.

#### 4.3. Semantic Segmentation

##### Settings. To evaluate the semantic segmentation per-

##### formance of InternImage, we initialize the backbone with


```
method crop #params #FLOPs mIoU mIoU
size (SS) (MS)
Swin-T [2] 5122 60M 945G 44.5 45.
ConvNeXt-T [21] 5122 60M 939G 46.0 46.
SLaK-T [29] 5122 65M 936G 47.6 −
InternImage-T (ours) 5122 59M 944G 47.9 48.
Swin-S [2] 5122 81M 1038G 47.6 49.
ConvNeXt-S [21] 5122 82M 1027G 48.7 49.
SLaK-S [29] 5122 91M 1028G 49.4 −
InternImage-S (ours) 5122 80M 1017G 50.1 50.
Swin-B [2] 5122 121M 1188G 48.1 49.
ConvNeXt-B [21] 5122 122M 1170G 49.1 49.
RepLKNet-31B [22] 5122 112M 1170G 49.9 50.
SLaK-B [29] 5122 135M 1172G 50.2 −
InternImage-B (ours) 5122 128M 1185G 50.8 51.
Swin-L‡[2] 6402 234M 2468G 52.1 53.
RepLKNet-31L‡[22] 6402 207M 2404G 52.4 52.
ConvNeXt-L‡[21] 6402 235M 2458G 53.2 53.
ConvNeXt-XL‡[21] 6402 391M 3335G 53.6 54.
InternImage-L‡(ours) 6402 256M 2526G 53.9 54.
InternImage-XL‡(ours) 6402 368M 3142G 55.0 55.
SwinV2-G#[16] 8962 3.00B − − 59.
InternImage-H#(ours) 8962 1.12B 3566G 59.9 60.
BEiT-3#[17] 8962 1.90B − − 62.
FD-SwinV2-G#[26] 8962 3.00B − − 61.
InternImage-H#(ours) +
Mask2Former [80]
```
```
8962 1.31B 4635G 62.5 62.
```
###### Table 5.Semantic segmentation performance on the ADE20K

###### validation set. The FLOPs are measured with 512×2048,

###### 640 ×2560, or 896×896 inputs according to the crop size. “SS”

###### and “MS” means single-scale and multi-scale testing, respectively.

##### pre-trained classification weights and train our models with

##### UperNet [81] on ADE20K [82] for 160k iterations and

##### compare fairly with previous CNN-based and transformer-

##### based backbones. To further reach top performance, we arm

##### InternImage-H with more advanced Mask2Former [80], and

##### adopt the same training settings in [17, 69].

##### Results. As shown in Table 5, when using UperNet

##### [81] for semantic segmentation, our InternImage consis-

##### tently outperforms prior arts [2, 21, 22, 29]. For exam-

##### ple, with almost the same parameter numbers and FLOPs,

##### our InternImage-B reports 50.8 mIoU on the ADE20K val,

##### which is outstanding from the strong counterparts such

##### as ConvNeXt-B (50.8vs.49.1) and RepLKNet-31B (50.

##### vs.49.9). Furthermore, our InternImage-H yields 60.3 MS

##### mIoU, which is better than SwinV2-G [16], while the pa-

##### rameter number is much smaller (1.12Bvs.3.00B).

##### It is worth noting that, when using Mask2Former [80]

##### and multi-scale testing, our InternImage-H achieves the best

##### mIoU of 62.9, higher than the current best BEiT-3 [17] on

##### the ADE20K benchmark. These results demonstrate that

##### the CNN-based foundation model can also enjoy the divi-

##### dends of massive data and challenge the leading position of

##### transformer-based models.

###### Figure 4.Model parameters and GPU memory usage of shared

###### weightsv.sunshared weights among convolution neurons.The

###### left vertical axis indicates the model parameters and the right one

###### indicates the GPU memory usage per image when the batch size

###### is 32 and the input image resolution is 224 × 224.

```
stage 1 stage 2
```
```
stage 3 stage 4
```
###### Figure 5. Visualization of sampling locations for different

###### groups at different stages. The blue star indicates the query point

###### (on the left sheep), and the dots with different colors indicate the

###### sampling locations of different groups.

#### 4.4. Ablation Study

##### Sharing weights among convolution neurons matters.

##### Large-scale models are sensitive to parameters and memory

##### cost of the core operator, due to hardware limitations. To

##### address this problem, we share weights among convolution

##### neurons of DCNv3. As shown in Fig. 4, we compare the pa-

##### rameters and memory cost of the models based on DCNv

##### with shared or unshared weights. We see that the parame-

##### ters and memory cost of models with unshared weights are

##### much higher than the shared one, especially for the -H scale,

##### the ratio of saved parameters and GPU memory is 42.0%

##### and 84.2%, respectively. As shown in Table 6, we also ex-

##### amine that the two models at -T scale have similar top-

##### accuracy on ImageNet (83.5vs. 83.6) and APbon COCO

##### (47.2vs. 47.4), even the model without shared weights has

##### 66.1% more parameters.

##### Multi-group spatial aggregation brings stronger fea-


```
sharedw multi-group softmax top-1 acc APb APm
733 83.6 47.4 42.
373 82.3 43.8 40.
337 65.7 38.7 35.
333 83.5 47.2 42.
```
###### Table 6. Ablation comparison of the three modifications in

###### DCNv3.These experiments are based on InternImage-T for clas-

###### sification and Mask R-CNN 1×schedule for detection.

##### tures.We introduce aggregation groups to allow our model

##### to learn information from different representation subspaces

##### like transformers [9]. As shown in Fig. 5, for the same

##### query pixel, the offsets from different groups are concen-

##### trated in different regions, resulting in hierarchical seman-

##### tic features. We also compare the performance of the model

##### with and without multiple groups. As reported in Table 6,

##### the model significantly drops 1.2 points on ImageNet and

##### 3.4 points on COCO val2017. In addition, we also see that

##### in the first two stages, the learned effective receptive field

##### (ERF) is relatively small, and as the model goes deeper (i.e.,

##### stages 3 and 4), the ERF increases to be global. This phe-

##### nomenon is different from ViTs [9, 10, 83] whose ERF is

##### usually global.

### 5. Conclusion & Limitations

##### We introduce InternImage, a new large-scale CNN-based

##### foundation model that can provide strong representations

##### for versatile vision tasks, such as image classification, ob-

##### ject detection, and semantic segmentation. We tune the flex-

##### ible DCNv2 operator to satisfy the requirement of foun-

##### dation models, and develop a series of blocks, stacking

##### and scaling rules centered on the core operator. Exten-

##### sive experiments on object detection and semantic segmen-

##### tation benchmarks verify that our InternImage can obtain

##### comparable or better performance than well-designed large-

##### scale vision transformers trained with massive data, show-

##### ing that CNN is also a considerable choice for large-scale

##### vision foundation model research. Nonetheless, latency re-

##### mains an issue for DCN-based operators adapting to down-

##### stream tasks with high-speed requirements. Also, large-

##### scale CNNs are still in their early stages of development,

##### and we hope InternImage can serve as a good starting point.


## Appendix

### A. Detailed Training Settings

##### In this section, we present the detailed training recipes

##### for image classification, object detection, and semantic seg-

##### mentation.

#### A.1. Settings for Backbone-Level Comparison

##### ImageNet image classification.The training details of

##### image classification on ImageNet [31] are shown in Table

##### 7, which are similar to common practices [2, 21, 58, 64] and

##### with some tweaks. To further explore the capability of our

##### model and match the large-scale private data used in previ-

##### ous methods [16, 20, 59], we adopt M3I Pre-training [60],

##### a unified pre-training approach available for both unla-

##### beled and weakly-labeled data, to pre-train InternImage-H

##### on a 427 million joint dataset of public Laion-400M [61],

##### YFCC-15M [62], and CC12M [63] for 30 epochs, and then

##### we fine-tune the model on ImageNet-1K for 20 epochs. For

##### the more detailed pre-training settings of InternImage-H,

##### please refer to M3I Pre-training [60].

##### COCO object detection. We verify the detection

##### performance of our InternImage on the COCO bench-

##### mark [32], on top of Mask R-CNN [70] and Cascade Mask

##### R-CNN [71]. For fair comparisons, we follow common

##### practices [2, 11] to initialize the backbone with pre-trained

##### classification weights, and train these models using a 1×

##### (12 epochs) or 3×(36 epochs) schedule by default. For 1×

##### schedule, the image is resized to have a shorter side of 800

##### pixels, while the longer side does not exceed 1,333 pixels.

##### During testing, the shorter side of the input image is fixed

##### to 800 pixels. For 3×schedule, the shorter side is resized

##### to 480−800 pixels, while the longer side does not exceed

##### 1,333 pixels. All these detection models are trained with

##### a batch size of 16 and optimized by AdamW [84] with an

##### initial learning rate of 1 × 10 −^4.

##### ADE20K semantic segmentation.We evaluate our In-

##### ternImage models on the ADE20K dataset [82], and initial-

##### ize them with the pre-trained classification weights. For

##### the InternImage-T/S/B models, we optimize them using

##### AdamW [84] with an initial learning rate of 6× 10 −^5 , and

##### 2 × 10 −^5 for InternImage-X/XL. The learning rate is de-

##### cayed following the polynomial decay schedule with a

##### power of 1.0. Following previous methods [2, 11, 21], the

##### crop size is set to 512 for InternImage-T/S/B, and 640 for

##### InternImage-L/XL. All segmentation models are trained us-

##### ing UperNet [81] with a batch size of 16 for 160k itera-

##### tions, and compared fairly with previous CNN-based and

##### transformer-based backbones.

#### A.2. Settings for System-Level Comparison

##### COCO object detection.For system-level comparison

##### with state-of-the-art large-scale detection models [16,17,26,

```
! 1
" 1 ,"′
```
###### Figure 6.Comparison of different stacking hyper-parameters.

###### Each square indicates the accuracy of the model determined by

###### hyperparameter, with the darker the color, the higher the accuracy.

##### 74, 78], we first initialize the InternImage-XL/H backbone

##### with the weights pre-trained on ImageNet-22K or the 427M

##### large-scale joint dataset, and double its parameters using the

##### composite techniques [78]. Then, we pre-train the model

##### along with the DINO [74] detector on the Objects365 [79]

##### for 26 epochs, with an initial learning rate of 2 × 10 −^4 and a

##### batch size of 256. The shorter size of input images is resized

##### to 600−1200 pixels during pre-training, and the learning

##### rate drops by 10 times at epoch 22. Finally, we fine-tune

##### these detectors on the COCO dataset for 12 epochs, where

##### the batch size is 64, and the initial learning rate is 5 × 10 −^5 ,

##### which drops by 10 times at the final epoch.

##### ADE20K semantic segmentation. To further reach

##### leading segmentation performance, we first initialize our

##### InternImage-H backbone with the pre-trained weights on

##### the 427M large-scale joint dataset, and arm it with the

##### state-of-the-art segmentation method Mask2Former [80].

##### We follow the same training settings in [17, 69],i.e.pre-

##### training and fine-tuning the model on COCO-Stuff [85] and

##### ADE20K [82] datasets both for 80k iterations, with a crop

##### size of 896 and an initial learning rate of 1× 10 −^5.

### B. Exploration of Hyper-parameters

#### B.1. Model Stacking

##### As discussed in Section 3.2, our model is constructed

##### in four stacking rules, and we further restrict the model

##### parameters to 30M for the origin model. We discretize

##### the stacking hyperparametersC 1 to{ 16 , 32 , 64 }, L 1 to

##### { 1 , 2 , 3 , 4 , 5 }, andC′to{ 16 , 32 }. AndL 2 is determined

##### by selecting the model size to approximately 30M. In this

##### way, we obtained 30 models by combining the three hyper-

##### parameters.

##### We adopt the training recipe listed in Table 7 to train

##### our -T models unless otherwise stated. Fig. 6 shows the

##### ImageNet-1K top-1 accuracy of these models under the


```
settings InternImage-T InternImage-S InternImage-B InternImage-L InternImage-XL InternImage-H
IN-1K pt IN-1K pt IN-1K pt IN-22K pt IN-1K ft IN-22K pt IN-1K ft IN-1K ft
input scale 224 224 224 192 384 192 384 224/
batch size 4096 4096 4096 4096 512 4096 512 512
optimizer AdamW AdamW AdamW AdamW AdamW AdamW AdamW AdamW
LR 4 × 10 −^34 × 10 −^34 × 10 −^31 × 10 −^32 × 10 −^51 × 10 −^32 × 10 −^52 × 10 −^5
LR schedule cosine cosine cosine cosine cosine cosine cosine cosine
weight decay 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.
warmup epochs 5 5 5 5 0 5 0 0
epochs 300 300 300 90 20 90 20 20
horizontal flip 3 3 3 33 33 3
random resized crop 3 3 3 33 33 3
auto augment 3 3 3 33 33 3
layer scale 7 3 3 33 33 3
mixup alpha 0.8 0.8 0.8 0.8 7 0.8 7 7
cutmix alpha 1.0 1.0 1.0 1.0 7 1.0 7 7
erasing prob. 0.25 0.25 0.25 0.25 7 0.25 7 7
color jitter 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.
label smoothingε 0.1 0.1 0.1 0.1 0.3 0.1 0.3 0.
dropout 7 7 7 77 77 7
drop path rate 0.1 0.4 0.5 0.1 0.1 0.2 0.2 0.
repeated aug 7 7 7 77 77 7
gradient clip 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.
loss CE CE CE CE CE CE CE CE
```
###### Table 7.Detailed training recipe for InternImage of different parameter scales on ImageNet [31].“CE” denotes the cross entropy

###### loss, “LR” denotes the learning rate. The training recipe follows common practices [2, 21, 58, 64] and has some tune-ups. “IN-1K pt”,

###### “IN-22K pt”, and “IN-1K ft” represent ImageNet-1K pre-training, ImageNet-22K pre-training, and ImageNet-1K fine-tuning, respectively.

##### same training settings, with darker green indicating higher

##### accuracy,i.e., models with stronger representational capa-

##### bility. WhenC′equals 16, models are generally higher than

##### that withC′of 32, andL 1 works best at 4, thanks to a rea-

##### sonable stacking ratio. A large number of channels allows

##### for more gain. Finally, through the above exploration exper-

##### iments, we determine our basic stacking hyper-parameter

##### (C 1 ,C′,L 1 ,L 3 )to(64, 16 , 4 ,18).

#### B.2. Model Scaling

##### In Section 3.2, we have shown the constraints on the

##### depth scaling factorαand the width scaling factorβ. Based

##### on this condition and the -T model (30M), we display rea-

##### sonable scaling possibilities for extending the -T model to

##### -B models (100M). As illustrated in Table 8, the first two

##### columns show the formulas forαandβ. The penultimate

##### column indicates model parameters, and the last column in-

##### dicates the ImageNet-1K top-1 accuracy of these models

##### after 300 training epochs.

##### It is worth noting that the model widthC 1 needs to be

##### divisible byC′. Therefore some adjustment is required in

##### determining the specific scaling parameters. This results in

##### a small fluctuation in the number of parameters, but this is

##### acceptable. Our exploratory experiments prove that when

##### (α,β)is set at(1. 09 , 1 .36)for the best performance. In

##### addition, the other size models -S/L/XL/H also confirmed

##### the effectiveness of our scaling rules.

```
scaling factors
α β #parameters top-1 accuracy (%)
1.03 1.40 118M 84.
1.06 1.38 95M 83.
1.09 1.36 97M 84.
1.12 1.34 105M 83.
1.15 1.32 95M 81.
```
###### Table 8.Comparison of different scaling factors. The default

###### setting is marked with a gray background.

```
kernel size #parameters FLOPs top-1 accuracy (%)
3 × 3 30M 5G 83.
5 × 5 37M 6G 83.
7 × 7 48M 8G 82.
```
###### Table 9.Comparison of different kernel sizes in our operator.

###### The default setting is marked with a gray background.

#### B.3. Kernel Size

##### As mentioned in Section 3.1, we argue 3×3 dynamic

##### sparse convolution is enough for the large receptive field.

##### Here, we explore the role played by the number of convo-

##### lutional neurons in the DCNv3 operator. Specifically, we

##### replaced the 3 × 3 kernel in the DCNv3 operator with the

##### 5 × 5 or 7 × 7 kernel. They are all trained by the -T train-

##### ing recipes (see Table 7) and validated on the ImageNet-1K

##### validation set. The results are shown in Table 9.

##### The results show that when enlarging the convolution

##### kernel, the parameters and FLOPs are followed by the


```
(a)ResNet101w/otraining
```
```
(b)ResNet101w/trainedmodel
```
```
(c)InternImage-Sw/otraining
```
```
(d)InternImage-Sw/trainedmodel
```
```
stage 1 stage 2 stage 3 stage 4
```
###### Figure 7. Visualization of the effective receptive field (ERF)

###### of different backbones.The activated pixel is at dog’s eye. (a)

###### and (b) shows the ERF of ResNet-101 [36] with (w/) and without

###### (w/o) training on ImageNet-1K [31], respectively. (c) and (d) are

###### the ERF of InternImage-B with (w/) and without (w/o) training on

###### ImageNet-1K.

##### surge, while the accuracy is not significantly improved

##### (83.5v.s83.6) or even decreased (83.5v.s82.8). These

##### results show that when the number of convolutional neu-

##### rons in a single layer increases, the model becomes more

##### difficult to optimize. This phenomenon is also confirmed

##### in RepLKNet [22], and it addresses this problem by re-

##### parameterizing [22] techniques, which might bring extra

##### time and memory costs in the training phase. In this work,

##### we avoid this problem by adopting the simple yet effective

##### 3 × 3 DCNv3 as InternImage’s core operator.

##### Fig. 7 shows the effective receptive fields (ERF) of

##### ResNet-101 [36] and InternImage-S. A wider distribution of

##### bright areas indicates a larger ERF. We uniformly activate

##### the input image at the dog’s eye, count the gradient map

##### of each block, aggregate by channel, and map back to the

##### input image. We see that the ERF of ResNet-101 [36] with-

##### out training is limited to a local area, while the fully trained

##### ResNet-101 still has an ERF around the eye, and the gradi-

##### ent amplitude is lower, and the distribution is more sparse.

##### Therefore, the area that ResNet-101 can effectively perceive

##### is very limited. For the InternImage-S without training, its

##### ERF is concentrated around the activation point. Since the

##### offset is not learned, its ERF is also very small in the last

##### two blocks. But after sufficient training, InternImage-L can

##### effectively perceive the information of the entire image in

##### the 3-rd and 4-th stages.

### C. Additional Downstream Tasks

#### C.1. Classification

##### iNaturalist 2018[98] is a read-word long-tailed dataset

##### containing 8142 fine-graned species. The dataset comprises

##### 437.5K training images and an imbalance factor of 500.

##### For this experiment, we initialize our InternImage-H model

##### with the pre-trained weights on the 427M large-scale joint

##### dataset, and fine-tune it on the training set of iNaturalist

##### 2018 for 100 epochs. We follow MetaFormer [86] to adopt

##### a resolution of 384×384 for fine-tuning, with the utilization

##### of meta information. Other training settings are the same as

##### the recipe for fine-tuning InternImage-H on ImageNet-1K,

##### as reported in Table 7. As a result, our method achieves

##### the state-of-the-art accuracy of 92.6 (see Table 10) on the

##### validation set of iNaturalist 2018, 3.9 points better than the

##### previous best model MetaFormer [86].

##### Places205[99] is a dataset containing 2.5 million im-

##### ages of 205 scene categories, which are dedicated to the

##### scene recognition task. The images in this dataset cover a

##### wide range of indoor and outdoor scenes, such as offices,

##### kitchens, forests, and beaches. We initialize our model

##### with pre-trained weights on a large-scale joint dataset,

##### consisting of 427 million images, and fine-tune it on the

##### Places205 training set. Other training settings are the same

##### as the recipe for fine-tuning InternImage-H on ImageNet-

##### 1K, as reported in Table 7. Our method achieves state-of-

##### the-art accuracy of 71.7 (see Table 10) on the validation

##### set of Places205, outperforming the previous best model

##### MixMIM-L [87] by 2.4 points.

##### Places365[100] is a dataset containing 1.8 million im-

##### ages of 365 scene categories, which are dedicated to the

##### scene recognition task. The images in this dataset cover a

##### wide range of indoor and outdoor scenes, such as airports,

##### bedrooms, deserts, and waterfalls. The specific pre-training

##### and fine-tuning strategies are the same as for Places205.

##### Our method achieves state-of-the-art accuracy of 61.2 (see

##### Table 10) on the validation set of Places365, outperform-

##### ing the previous best model SWAG [88] by 0.5 points. The

##### Places365 dataset provides a more fine-grained classifica-

##### tion task compared to Places205, allowing our model to

##### learn more subtle differences between similar scenes.

#### C.2. Object Detection

##### LVIS v1.0[101] is a large-scale vocabulary dataset for

##### object detection and instance segmentation tasks, which

##### contains 1203 categories in 164k images. For this dataset,

##### we initialize our InternImage-H with the Objects365 [79]


```
method classification semantic segmentation
iNaturalist2018 Places205 Places365 COCO-Stuff-10K Pascal Context Cityscapes (val) Cityscapes (test) NYU Depth V
previous best 88.7a 69.3b 60.7c 54.2d 68.2d 86.9e 85.2d 56.9f
InternImage-H 92.6 (+3.9) 71.7 (+2.4) 61.2 (+0.5) 59.6 (+5.4) 70.3 (+2.1) 87.0 (+0.1) 86.1 (+0.9) 68.1 (+11.2)
```
```
method
```
```
object detection
LVIS (minival) LVIS (val) VOC2007 VOC2012 OpenImages CrowdHuman BDD100K
previous best 59.8g 62.2h 89.3i 92.9j 72.2k 94.1l 35.6m
InternImage-H 65.8 (+6.0) 63.2 (+1.0) 94.0 (+4.7) 97.2 (+4.3) 74.1 (+1.9) 97.2 (+3.1) 38.8 (+3.2)
```
###### Table 10.Summary of InternImage-H performance on various mainstream vision benchmarks. a: MetaFormer [86]. b: MixMIM-

###### L [87]. c: SWAG [88]. d: ViT-Adapter [69]. e: PSA [89]. f: CMX-B5 [90]. g: GLIPv2 [91]. h: EVA [92]. i: Cascade Eff-B

###### NAS-FPN [93]. j: ATLDETv2 [94]. k: OpenImages 2019 competition 1st[95]. l: Iter-Deformable-DETR [96]. m: PP-YOLOE [97].

##### pre-trained weights, then fine-tune it on the training set of

##### LVIS v1.0. Here, we report the box AP (i.e., APb) with

##### multi-scale testing on the minival set and the val set, re-

##### spectively. As shown in Table 10, our InternImage-H cre-

##### ates a new record of 65.8 APbon the LVIS minival, and

##### 63.2 APbon the LVIS val, outperforming previous state-of-

##### the-art methods by clear margins.

##### Pascal VOC[102] contains 20 object classes, which

##### has been widely used as a benchmark for object detection

##### tasks. We adopt this dataset to further evaluate the detec-

##### tion performance of our model. Specifically, we employ

##### the Objects365 [79] pre-trained weights to initialize our

##### InternImage-H, and fine-tune it on the trainval set of Pas-

##### cal VOC 2007 and Pascal VOC 2012 following previous

##### method [93]. As shown in Table 10, on the Pascal VOC

##### 2007 test set, our InternImage-H yields 94.0 AP^50 with

##### single-scale testing, which is 4.7 points better than previous

##### best Cascade Eff-B7 NAS-FPN [93]. On the Pascal VOC

##### 2012 test set, our method achieves 97.2 mAP, 4.3 points

##### higher than the best record on the official leaderboard [94].

##### OpenImages v6[103] is a dataset of about 9 million

##### images with 16M bounding boxes for 600 object classes

##### on 1.9 million images dedicated to the object detection

##### task, which are very diverse and often embrace complex

##### scenes with multiple objects (8.3 per image on average).

##### For this dataset, we use the same settings as the previous

##### two datasets. In addition, we follow [95] to use the class-

##### aware sampling during fine-tuning. As reported in Table 10,

##### our InternImage-H yields 74.1 mAP, achieving 1.9 mAP im-

##### provement compared to the previous best results [95].

##### CrownHuman[104] is a benchmark dataset to better

##### evaluate detectors in crowd scenarios. The CrowdHuman

##### dataset is large, rich-annotated and contains high diversity.

##### CrowdHuman contains 15000, 4370 and 5000 images for

##### training, validation, and testing, respectively. There are a

##### total of 470K human instances from train and validation

##### subsets and 23 persons per image, with various kinds of oc-

##### clusions in the dataset. We used the same training setup

##### as for the previous dataset. Our pre-trained model reached

##### optimal performance in 3750 iterations, exceeding the pre-

##### vious best model Iter-Deformable-DETR [96] by 3.1 AP.

##### BDD100K[105] is a dataset of around 100K high-

##### resolution images with diverse weather and lighting con-

##### ditions, containing 10 object categories, including pedestri-

##### ans, cars, buses, and bicycles, dedicated to the object de-

##### tection task. The images in this dataset are captured from

##### a moving vehicle, simulating real-world scenarios. For this

##### experiment, we initialize our InternImage-H model with the

##### pre-trained weights on the 427M joint dataset and fine-tune

##### it on the BDD100K training set for 12 epochs. As re-

##### ported in Table 10, our InternImage-H achieves 38.8 mAP

##### on the validation set, which is the state-of-the-art perfor-

##### mance, surpassing the previous best model by 3.2 mAP. Our

##### method demonstrates superior performance in detecting ob-

##### jects in real-world driving scenarios, which can benefit au-

##### tonomous driving and intelligent transportation systems.

#### C.3. Semantic Segmentation

##### COCO-Stuff[85] includes the images from the COCO

##### [32] dataset for semantic segmentation, spanning over 171

##### categories. Specifically, COCO-Stuff-164K is the full set

##### that contains all 164k images, while COCO-Stuff-10K is a

##### subset of the -164K that splits into 9,000 and 1,000 images

##### for training and testing. Here, we equip our InternImage-

##### H with the advanced Mask2Former [80], and pre-train the

##### model on the COCO-Stuff-164K for 80k iterations. Then

##### we fine-tune it on the COCO-Stuff-10K for 40k iterations

##### and report the multi-scale mIoU. The crop size is set to

##### 512 ×512 in this experiment. As shown in Table 10, our

##### model achieves 59.6 MS mIoU on the test set, outperform-

##### ing the previous best ViT-Adapter [69] by 5.4 mIoU.

##### Pascal Context[106] contains 59 semantic classes. It

##### is divided into 4,996 images for training and 5,104 images

##### for testing. For this dataset, we also employ Mask2Former

##### with our InternImage-H, and follow the training settings

##### in [69]. Specifically, we first load the classification pre-

##### trained weights to initialize the model, then fine-tune it on

##### the training set of Pascal Context for 40k iterations. The

##### crop size is set to 480×480 in this experiment. As shown in

##### Table 10, our method reports 70.3 MS mIoU on the test set,

##### which is 2.1 points better than ViT-Adapter [69].

##### Cityscapes[107] is a high-resolution dataset recorded


```
method #paramsscaleFLOPsacc (%)throughput (img/s)
InternImage-B 2242 16G 84.9 775
(ours)
```
```
97M
8002 206G − 54
InternImage-B- 2242 24G − 311
DCNv2 [28] 146M 8002 313G − 16
```
```
ConvNeXt-B [21] 88M
```
```
2242 15G 83.8 881
8002 196G − 58
```
```
RepLKNet-B [22] 79M^224
```
(^2) 15G 83.5 884
8002 198G − 21
DAT-B [21] 88M
2242 16G 84.0 661
8002 194G − 24

###### Table 11.Throughput comparison of different models under

###### different input resolutions.“#params” denotes the number of pa-

###### rameters. “acc” represents the top-1 accuracy on the ImageNet-1K

###### validation set. The throughputs of 224×224 and 800×800 input

###### resolutions are tested with the batch size of 256 and 2 respectively,

###### using a single A100 GPU.

##### in street scenes including 19 classes. In this experiment,

##### we use Mask2Former [80] as the segmentation framework.

##### Following common practices [69, 83, 108], we first pre-train

##### on Mapillary Vistas [109] and then fine-tune on Cityscapes

##### for 80k iterations, respectively. The crop size is set to

##### 1024 ×1024 in this experiment. As shown in Table 10, our

##### InternImage-H achieves 87.0 MS mIoU on the validation

##### set, and 86.1 MS mIoU on the test set.

##### NYU Depth V2[110] comprises of 1449 RGB-D im-

##### ages, each with a size of 640×480. These images are di-

##### vided into 795 training and 654 testing images, each with

##### annotations on 40 semantic categories. We adopt the same

##### training settings as we used when fine-tuning on Pascal

##### Context. As shown in Table 10, our method achieves a big

##### jump to 68.1 MS mIoU on the validation set, which is 11.

##### points better than CMX-B5 [90].

### D. Throughput Analysis

##### In this section, we benchmark the throughput of our In-

##### ternImage with counterparts, including a variant equipped

##### with DCNv2 [28], ConvNext [21], RepLKNet [22], and a

##### vision transformer with deformable attention (DAT) [46].

##### As shown in Table 11, compared to the variant with DCNv

##### [28], our model enjoys better parameter-efficient and sig-

##### nificantly faster inference speed under both 224×224 and

##### 800 ×800 input resolutions. Compared to RepLKNet-B [22]

##### and DAT-B [46], our model has a throughput advantage at

##### a high input resolution (i.e., 800×800). This resolution is

##### widely used in dense prediction tasks such as object detec-

##### tion. Compared with ConvNeXt [21], despite the through-

##### put gap due to DCN-based operators, our model still has an

##### accuracy advantage (84.9vs. 83.8), and we are also looking

##### for an efficient DCN to make our model more suitable for

##### downstream tasks that require high efficiency.

```
86
```
```
88
```
```
90
```
```
92
```
```
94
```
```
96
```
```
98
```
```
100
```
```
0 16 32 48 64
```
```
consistency
```
```
thenumberoftranslationpixels
```
```
ConvNeXt
PVTv
Swin
Ours
77
```
```
78
```
```
79
```
```
80
```
```
81
```
```
82
```
```
83
```
```
84
```
```
0 16 32 48 64
```
```
top
```
-^1 accuracy

```
thenumberoftranslationpixels
```
```
ConvNeXt
PVTv
Swin
Ours
```
```
80
```
```
84
```
```
88
```
```
92
```
```
96
```
```
100
```
```
0 15 30 45
```
```
consistency
```
```
the angle of image rotation
```
```
ConvNeXt
PVTv
Swin
Ours
72
```
```
74
```
```
76
```
```
78
```
```
80
```
```
82
```
```
84
```
```
0 15 30 45
```
```
top
```
-^1

```
ccuracy
```
```
the angle of image rotation
```
```
ConvNeXt
PVTv
Swin
Ours
```
```
5
```
```
15
```
```
25
```
```
35
```
```
45
```
```
55
```
```
0.25 0.75 1.25 1.75 2.25 2.
```
```
consistency
```
```
scalefactor
```
```
ConvNeXt
PVTv
Swin
Ours
15
```
```
20
```
```
25
```
```
30
```
```
35
```
```
40
```
```
45
```
```
50
```
```
0.25 0.75 1.25 1.75 2.25 2.
```
```
bounding
```
```
box
```
```
mAP
```
```
scalefactor
```
```
ConvNeXt
PVTv
Swin
Ours
```
###### Figure 8.Comparison of robust evaluation of different meth-

###### ods. These results show that our model has better robustness in

###### terms of translation, rotation, and input resolution.

### E. Robustness Evaluation on ImageNet

##### In this section, we evaluate the robustness of differ-

##### ent models under different transformations (see Fig. 8).

##### We consider translation, rotation, and scaling to evalu-

##### ate. The models we choose for comparison include a

##### convolutional model (ConvNeXt-T [21]), a local attention-

##### based model (Swin-T [2]), a global attention-based model

##### (PVTv2-B2 [11]), and our InternImage-T.

#### E.1. Translation Invariance

##### Translation invariance describes the capability of the

##### model to retain the original output when the input image

##### is translated. We evaluate the translation invariance in the

##### classification task by dithering the image from 0 to 64 pix-

##### els. The invariance is measured by the probability that the

##### model predicts the same label when the same input image

##### is translated. The first row of Fig. 8 indicates our Intern-

##### Imagehas the translation invariance of the different meth-

##### ods. It is evident that the robustness of the four mod-

##### els to translation is shown as our method is the best, fol-

##### lowed by convolution-based ConvNeXt, followed by global

##### attention-based PVTv2, and the worst local attention-based

##### Swin transformer.


#### E.2. Rotation Invariance

##### To evaluate the rotation invariance of the classification

##### task, we rotate the image from 0 ◦to 45 ◦in steps of 5 ◦. In

##### a similar way to translation invariance, the predicted con-

##### sistency under different rotation angles is used to evaluate

##### the rotational invariance. From the second row of Fig. 8,

##### we found that the consistency performance of all models is

##### comparable in the small angle phase. However, at large-

##### angle rotation (i.e.,> 10 ◦), our model is clearly superior to

##### the other models.

#### E.3. Scaling Invariance

##### We evaluate the scaling invariance on object detection.

##### The scaling factor of the input image varies from 0.25 to 3.

##### in steps of 0.25. Detection consistency is defined as the in-

##### variance metric for the detection task. The predicted boxes

##### on the scaled images are first converted back to the original

##### resolution, and then the predicted boxes at the original res-

##### olution are used as the ground truth boxes to calculate the

##### box mAP. As seen in the last row of Fig. 8, we can observe

##### that all methods of our experiments are sensitive to down-

##### scaling. And they show invariance comparable to the input

##### at small resolutions. Our method performs better when scal-

##### ing up the images. Both box consistency and bounding box

##### mAP are better than the others.

#### E.4. How Hungry the Model is for Data Scale?

##### In order to verify the robustness of the model to the

##### data scale. We uniformly sampled the ImageNet-1K data

##### to obtain 1%, 10%, and 100% data, respectively. And

##### we chose ResNet-50 [36], ConvNeXt-T [21], Swin-T [2],

##### InternImage-T-dattn and our InternImage-T to conduct 300

##### rounds of training experiments on these data. The exper-

##### imental settings are consistent with Table 7. The exper-

##### imental results can be viewed in Table 12. We see that

##### ResNet [36] performs best on the 1% and 10% data (12.2%

##### & 57.5%), benefiting from its inductive biases. But its

##### upper limitation is low (80.4%) when the data is suffi-

##### cient. Swin-T fails completely in 1% datasets and shows

##### good performance only on the 100% dataset. The proposed

##### InternImage-T has strong robustness not only on 1% and

##### 10% data (5.9% and 56.0%) but also on full data (83.5%),

##### which is consistently better than the InternImage-T vari-

##### ant with deformable attention (dattn) and ConvNeXt [21].

##### These results indicate the robustness of our model with re-

##### spect to the data scale.

### References

###### [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob

###### Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,

###### and Illia Polosukhin. Attention is all you need.Adv. Neural

###### Inform. Process. Syst., 30, 2017. 1, 2, 4, 5

```
method 1% 10% 100%
ResNet-50 [36] 12.2 57.5 80.
ConvNeXt-T [21] 8.4 52.6 82.
Swin-T [2] failed 12.1 81.
InternImage-T-dattn [56] 4.1 49.9 81.
InternImage-T (ours) 5.9 56.0 83.
```
###### Table 12.Accuracy of different models at different data scales.

###### “InternImage-dattn” refers to the model variant equipped with de-

###### formable attention [56].

###### [2] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng

###### Zhang, Stephen Lin, and Baining Guo. Swin transformer:

###### Hierarchical vision transformer using shifted windows. In

###### Int. Conf. Comput. Vis., pages 10012–10022, 2021. 1, 2, 3,

###### 5, 6, 7, 8, 10, 11, 14, 15

###### [3] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick

###### LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-

###### lm: Training multi-billion parameter language models us-

###### ing model parallelism. arXiv preprint arXiv:1909.08053,

###### 2019. 1

###### [4] Alec Radford, Jeffrey Wu, Rewon Child, David Luan,

###### Dario Amodei, Ilya Sutskever, et al. Language models

###### are unsupervised multitask learners.OpenAI blog, 1(8):9,

###### 2019. 1

###### [5] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine

###### Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,

###### and Peter J Liu. Exploring the limits of transfer learning

###### with a unified text-to-text transformer.Journal of Machine

###### Learning Research, 21:1–67, 2020. 1

###### [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub-

###### biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakan-

###### tan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

###### Language models are few-shot learners. Adv. Neural In-

###### form. Process. Syst., 33:1877–1901, 2020. 1

###### [7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,

###### Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul

###### Barham, Hyung Won Chung, Charles Sutton, Sebastian

###### Gehrmann, et al. Palm: Scaling language modeling with

###### pathways.arXiv preprint arXiv:2204.02311, 2022. 1

###### [8] William Fedus, Barret Zoph, and Noam Shazeer. Switch

###### transformers: Scaling to trillion parameter models with

###### simple and efficient sparsity.Journal of Machine Learning

###### Research, 23(120):1–39, 2022. 1

###### [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,

###### Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,

###### Mostafa Dehghani, Matthias Minderer, Georg Heigold,

###### Sylvain Gelly, et al. An image is worth 16x16 words: Trans-

###### formers for image recognition at scale. InInt. Conf. Learn.

###### Represent., 2020. 1, 2, 3, 5, 7, 9

###### [10] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao

###### Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.

###### Pyramid vision transformer: A versatile backbone for dense

###### prediction without convolutions. InInt. Conf. Comput. Vis.,

###### pages 568–578, 2021. 1, 3, 5, 6, 9


###### [11] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao

###### Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvt

###### v2: Improved baselines with pyramid vision transformer.

###### Computational Visual Media, 8(3):415–424, 2022. 1, 2, 3,

###### 5, 6, 7, 10, 14

###### [12] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming

###### Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining

###### Guo. Cswin transformer: A general vision transformer

###### backbone with cross-shaped windows. IEEE Conf. Com-

###### put. Vis. Pattern Recog., pages 12124–12134, 2022. 1, 6

###### [13] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu,

###### Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing

###### convolutions to vision transformers. InInt. Conf. Comput.

###### Vis., pages 22–31, 2021. 1

###### [14] Alaaeldin Ali, Hugo Touvron, Mathilde Caron, Piotr Bo-

###### janowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Na-

###### talia Neverova, Gabriel Synnaeve, Jakob Verbeek, et al.

###### Xcit: Cross-covariance image transformers. Adv. Neural

###### Inform. Process. Syst., 34, 2021. 1

###### [15] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu,

###### and Yunhe Wang. Transformer in transformer.Adv. Neural

###### Inform. Process. Syst., 34, 2021. 1

###### [16] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie,

###### Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong,

###### et al. Swin transformer v2: Scaling up capacity and res-

###### olution.Adv. Neural Inform. Process. Syst., pages 12009–

###### 12019, 2022. 1, 2, 3, 6, 7, 8, 10

###### [17] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhil-

###### iang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mo-

###### hammed, Saksham Singhal, Subhojit Som, et al. Image as a

###### foreign language: Beit pretraining for all vision and vision-

###### language tasks.arXiv preprint arXiv:2208.10442, 2022. 1,

###### 3, 7, 8, 10

###### [18] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim

###### Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel ́

###### Keysers, and Neil Houlsby. Scaling vision with sparse

###### mixture of experts. Adv. Neural Inform. Process. Syst.,

###### 34:8583–8595, 2021. 1, 2

###### [19] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and

###### Lucas Beyer. Scaling vision transformers. InIEEE Conf.

###### Comput. Vis. Pattern Recog., pages 12104–12113, 2022. 1,

###### 2, 3

###### [20] Zihang Dai, Hanxiao Liu, Quoc V Le, and Mingxing Tan.

###### Coatnet: Marrying convolution and attention for all data

###### sizes.Adv. Neural Inform. Process. Syst., 34:3965–3977,

###### 2021. 1, 2, 3, 6, 10

###### [21] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feicht-

###### enhofer, Trevor Darrell, and Saining Xie. A convnet for the

###### 2020s.arXiv preprint arXiv:2201.03545, 2022. 2, 3, 5, 6,

###### 7, 8, 10, 11, 14, 15

###### [22] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and

###### Guiguang Ding. Scaling up your kernels to 31x31: Re-

###### visiting large kernel design in cnns. InIEEE Conf. Comput.

###### Vis. Pattern Recog., pages 11963–11975, 2022. 2, 3, 5, 6,

###### 7, 8, 12, 14

###### [23] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen

###### Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan.

###### Metaformer is actually what you need for vision. InIEEE

###### Conf. Comput. Vis. Pattern Recog., pages 10819–10829,

###### 2022. 2

###### [24] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.

###### Layer normalization. arXiv preprint arXiv:1607.06450,

###### 2016. 2, 4, 5

###### [25] Dan Hendrycks and Kevin Gimpel. Gaussian error lin-

###### ear units (gelus). arxiv. arXiv preprint arXiv:1606.08415,

###### 2016. 2, 5

###### [26] Yixuan Wei, Han Hu, Zhenda Xie, Zheng Zhang, Yue Cao,

###### Jianmin Bao, Dong Chen, and Baining Guo. Contrastive

###### learning rivals masked image modeling in fine-tuning via

###### feature distillation.arXiv preprint arXiv:2205.14141, 2022.

###### 2, 7, 8, 10

###### [27] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong

###### Zhang, Han Hu, and Yichen Wei. Deformable convolu-

###### tional networks. InInt. Conf. Comput. Vis., pages 764–773,

###### 2017. 2, 3

###### [28] Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. De-

###### formable convnets v2: More deformable, better results. In

###### IEEE Conf. Comput. Vis. Pattern Recog., pages 9308–9316,

###### 2019. 2, 3, 14

###### [29] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Xuxi Chen,

###### Qiao Xiao, Boqian Wu, Mykola Pechenizkiy, Decebal Mo-

###### canu, and Zhangyang Wang. More convnets in the 2020s:

###### Scaling up kernels beyond 51x51 using sparsity. arXiv

###### preprint arXiv:2207.03620, 2022. 2, 6, 8

###### [30] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and

###### Lucas Beyer. Scaling vision transformers. InIEEE Conf.

###### Comput. Vis. Pattern Recog., pages 12104–12113, 2022. 2,

###### 6

###### [31] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,

###### and Li Fei-Fei. Imagenet: A large-scale hierarchical im-

###### age database. InIEEE Conf. Comput. Vis. Pattern Recog.,

###### pages 248–255, 2009. 2, 5, 6, 10, 11, 12

###### [32] Tsung-Yi Lin, Michael Maire, Serge Belongie, James

###### Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and ́

###### C Lawrence Zitnick. Microsoft coco: Common objects in

###### context. InEur. Conf. Comput. Vis., pages 740–755, 2014.

###### 2, 6, 10, 13

###### [33] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.

###### Imagenet classification with deep convolutional neural net-

###### works.Communications of the ACM, 60(6):84–90, 2017. 2,

###### 4

###### [34] Karen Simonyan and Andrew Zisserman. Very deep convo-

###### lutional networks for large-scale image recognition.arXiv

###### preprint arXiv:1409.1556, 2014. 3

###### [35] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet,

###### Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent

###### Vanhoucke, and Andrew Rabinovich. Going deeper with

###### convolutions. InIEEE Conf. Comput. Vis. Pattern Recog.,

###### pages 1–9, 2015. 3


###### [36] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

###### Deep residual learning for image recognition. InIEEE

###### Conf. Comput. Vis. Pattern Recog., pages 770–778, 2016.

###### 3, 5, 12, 15

###### [37] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, ́

###### and Kaiming He. Aggregated residual transformations for

###### deep neural networks. InIEEE Conf. Comput. Vis. Pattern

###### Recog., pages 1492–1500, 2017. 3

###### [38] Mingxing Tan and Quoc Le. Efficientnet: Rethinking

###### model scaling for convolutional neural networks. InInter-

###### national Conference on Machine Learning., pages 6105–

###### 6114. PMLR, 2019. 3, 5

###### [39] Mingxing Tan and Quoc Le. Efficientnetv2: Smaller mod-

###### els and faster training. InInternational Conference on Ma-

###### chine Learning., pages 10096–10106. PMLR, 2021. 3

###### [40] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry

###### Kalenichenko, Weijun Wang, Tobias Weyand, Marco An-

###### dreetto, and Hartwig Adam. Mobilenets: Efficient convolu-

###### tional neural networks for mobile vision applications.arXiv

###### preprint arXiv:1704.04861, 2017. 3

###### [41] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and

###### Guiguang Ding. Scaling up your kernels to 31x31: Re-

###### visiting large kernel design in cnns. InIEEE Conf. Comput.

###### Vis. Pattern Recog., pages 11963–11975, 2022. 3

###### [42] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Xuxi Chen,

###### Qiao Xiao, Boqian Wu, Mykola Pechenizkiy, Decebal Mo-

###### canu, and Zhangyang Wang. More convnets in the 2020s:

###### Scaling up kernels beyond 51x51 using sparsity. arXiv

###### preprint arXiv:2207.03620, 2022. 3

###### [43] Yongming Rao, Wenliang Zhao, Yansong Tang, Jie Zhou,

###### Ser-Nam Lim, and Jiwen Lu. Hornet: Efficient high-order

###### spatial interactions with recursive gated convolutions.arXiv

###### preprint arXiv:2207.14284, 2022. 3, 6, 7

###### [44] Qi Han, Zejia Fan, Qi Dai, Lei Sun, Ming-Ming Cheng, Ji-

###### aying Liu, and Jingdong Wang. On the connection between

###### local attention and dynamic depth-wise convolution. InInt.

###### Conf. Learn. Represent., 2021. 3

###### [45] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang,

###### and Hao Ma. Linformer: Self-attention with linear com-

###### plexity.arXiv preprint arXiv:2006.04768, 2020. 3

###### [46] Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao

###### Huang. Vision transformer with deformable attention. In

###### IEEE Conf. Comput. Vis. Pattern Recog., pages 4794–4803,

###### 2022. 3, 5, 14

###### [47] Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas,

###### Niki Parmar, Blake Hechtman, and Jonathon Shlens. Scal-

###### ing local self-attention for parameter efficient visual back-

###### bones. InIEEE Conf. Comput. Vis. Pattern Recog., pages

###### 12894–12904, 2021. 3

###### [48] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B

###### Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec

###### Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for

###### neural language models.arXiv preprint arXiv:2001.08361,

###### 2020. 3

###### [49] Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong

###### Wang, and Lu Yuan. Davit: Dual attention vision trans-

###### formers.arXiv preprint arXiv:2204.03645, 2022. 3

###### [50] Xizhou Zhu, Dazhi Cheng, Zheng Zhang, Stephen Lin, and

###### Jifeng Dai. An empirical study of spatial attention mecha-

###### nisms in deep networks. InInt. Conf. Comput. Vis., pages

###### 6688–6697, 2019. 3

###### [51] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos,

###### Kevin Murphy, and Alan L Yuille. Deeplab: Semantic im-

###### age segmentation with deep convolutional nets, atrous con-

###### volution, and fully connected crfs. IEEE Trans. Pattern

###### Anal. Mach. Intell., 40(4):834–848, 2017. 3

###### [52] L-CCGP Florian and Schroff Hartwig Adam. Rethinking

###### atrous convolution for semantic image segmentation. In

###### IEEE Conf. Comput. Vis. Pattern Recog., volume 6, 2017.

###### 3

###### [53] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Flo-

###### rian Schroff, and Hartwig Adam. Encoder-decoder with

###### atrous separable convolution for semantic image segmen-

###### tation. InEur. Conf. Comput. Vis., pages 801–818, 2018.

###### 3

###### [54] Yann LeCun, Bernhard Boser, John S Denker, Donnie

###### Henderson, Richard E Howard, Wayne Hubbard, and

###### Lawrence D Jackel. Backpropagation applied to handwrit-

###### ten zip code recognition. Neural Computation, 1(4):541–

###### 551, 1989. 3

###### [55] Franc ̧ois Chollet. Xception: Deep learning with depthwise

###### separable convolutions. InIEEE Conf. Comput. Vis. Pattern

###### Recog., pages 1251–1258, 2017. 4

###### [56] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang

###### Wang, and Jifeng Dai. Deformable detr: Deformable trans-

###### formers for end-to-end object detection. arXiv preprint

###### arXiv:2010.04159, 2020. 5, 15

###### [57] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin

###### Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei

###### Wang, and Tieyan Liu. On layer normalization in the trans-

###### former architecture. InInternational Conference on Ma-

###### chine Learning., pages 10524–10533. PMLR, 2020. 5

###### [58] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco

###### Massa, Alexandre Sablayrolles, and Herv ́e J ́egou. Training

###### data-efficient image transformers & distillation through at-

###### tention. InInternational Conference on Machine Learning.,

###### pages 10347–10357, 2021. 6, 10, 11

###### [59] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella,

###### Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang,

###### Boxin Li, Chunyuan Li, et al. Florence: A new

###### foundation model for computer vision. arXiv preprint

###### arXiv:2111.11432, 2021. 6, 7, 10

###### [60] Weijie Su, Xizhou Zhu, Chenxin Tao, Lewei Lu, Bin Li,

###### Gao Huang, Yu Qiao, Xiaogang Wang, Jie Zhou, and

###### Jifeng Dai. Towards all-in-one pre-training via maxi-

###### mizing multi-modal mutual information. arXiv preprint

###### arXiv:2211.09807, 2022. 6, 10


###### [61] Christoph Schuhmann, Richard Vencu, Romain Beaumont,

###### Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo

###### Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-

###### 400m: Open dataset of clip-filtered 400 million image-text

###### pairs.arXiv preprint arXiv:2111.02114, 2021. 6, 10

###### [62] Bart Thomee, David A Shamma, Gerald Friedland, Ben-

###### jamin Elizalde, Karl Ni, Douglas Poland, Damian Borth,

###### and Li-Jia Li. Yfcc100m: The new data in multimedia re-

###### search.Communications of the ACM, 59(2):64–73, 2016.

###### 6, 10

###### [63] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu

###### Soricut. Conceptual 12m: Pushing web-scale image-text

###### pre-training to recognize long-tail visual concepts. InIEEE

###### Conf. Comput. Vis. Pattern Recog., pages 3558–3568, 2021.

###### 6, 10

###### [64] Hugo Touvron, Matthieu Cord, and Herv ́e J ́egou. Deit iii:

###### Revenge of the vit.arXiv preprint arXiv:2204.07118, 2022.

###### 6, 10, 11

###### [65] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang

###### Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao. Focal self-

###### attention for local-global interactions in vision transform-

###### ers.arXiv preprint arXiv:2107.00641, 2021. 6

###### [66] Ismail Khalfaoui Hassani, Thomas Pellegrini, and Tim-

###### oth ́ee Masquelier. Dilated convolution with learnable spac-

###### ings.arXiv preprint arXiv:2112.03740, 2021. 6

###### [67] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan

###### Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.

###### Big transfer (bit): General visual representation learning.

###### InEur. Conf. Comput. Vis., pages 491–507. Springer, 2020.

###### 6

###### [68] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V

###### Le. Self-training with noisy student improves imagenet

###### classification. InIEEE Conf. Comput. Vis. Pattern Recog.,

###### pages 10687–10698, 2020. 6

###### [69] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong

###### Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for

###### dense predictions.arXiv preprint arXiv:2205.08534, 2022.

###### 7, 8, 10, 13, 14

###### [70] Kaiming He, Georgia Gkioxari, Piotr Doll ́ar, and Ross Gir-

###### shick. Mask r-cnn. InInt. Conf. Comput. Vis., pages 2961–

###### 2969, 2017. 7, 10

###### [71] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: high

###### quality object detection and instance segmentation.IEEE

###### Trans. Pattern Anal. Mach. Intell., 43(5):1483–1498, 2019.

###### 7, 10

###### [72] Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen,

###### Mengchen Liu, Lu Yuan, and Lei Zhang. Dynamic head:

###### Unifying object detection heads with attentions. InIEEE

###### Conf. Comput. Vis. Pattern Recog., pages 7373–7382, 2021.

###### 7

###### [73] Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan

###### Wang, Fangyun Wei, Xiang Bai, and Zicheng Liu. End-to-

###### end semi-supervised object detection with soft teacher. In

###### Int. Conf. Comput. Vis., pages 3060–3069, 2021. 7

###### [74] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun

###### Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr

###### with improved denoising anchor boxes for end-to-end ob-

###### ject detection.arXiv preprint arXiv:2203.03605, 2022. 7,

###### 10

###### [75] Jianwei Yang, Chunyuan Li, and Jianfeng Gao. Focal mod-

###### ulation networks.arXiv preprint arXiv:2203.11926, 2022.

###### 7

###### [76] Qiang Chen, Xiaokang Chen, Gang Zeng, and Jingdong

###### Wang. Group detr: Fast training convergence with de-

###### coupled one-to-many label assignment. arXiv preprint

###### arXiv:2207.13085, 2022. 7

###### [77] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He.

###### Exploring plain vision transformer backbones for object de-

###### tection.arXiv preprint arXiv:2203.16527, 2022. 7

###### [78] Tingting Liang, Xiaojie Chu, Yudong Liu, Yongtao Wang,

###### Zhi Tang, Wei Chu, Jingdong Chen, and Haibin Ling. Cb-

###### net: A composite backbone network architecture for object

###### detection.IEEE Trans. Image Process., 2022. 7, 10

###### [79] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang

###### Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A

###### large-scale, high-quality dataset for object detection. InInt.

###### Conf. Comput. Vis., pages 8430–8439, 2019. 7, 10, 12, 13

###### [80] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-

###### der Kirillov, and Rohit Girdhar. Masked-attention mask

###### transformer for universal image segmentation. arXiv

###### preprint arXiv:2112.01527, 2021. 8, 10, 13, 14

###### [81] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and

###### Jian Sun. Unified perceptual parsing for scene understand-

###### ing. InEur. Conf. Comput. Vis., pages 418–434, 2018. 8,

###### 10

###### [82] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela

###### Barriuso, and Antonio Torralba. Scene parsing through

###### ade20k dataset. InIEEE Conf. Comput. Vis. Pattern Recog.,

###### pages 633–641, 2017. 8, 10

###### [83] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar,

###### Jose M Alvarez, and Ping Luo. Segformer: Simple and ef-

###### ficient design for semantic segmentation with transformers.

###### Adv. Neural Inform. Process. Syst., 34, 2021. 9, 14

###### [84] Ilya Loshchilov and Frank Hutter. Decoupled weight decay

###### regularization.arXiv preprint arXiv:1711.05101, 2017. 10

###### [85] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-

###### stuff: Thing and stuff classes in context. InIEEE Conf.

###### Comput. Vis. Pattern Recog., pages 1209–1218, 2018. 10,

###### 13

###### [86] Qishuai Diao, Yi Jiang, Bin Wen, Jia Sun, and Zehuan

###### Yuan. Metaformer: A unified meta framework for fine-

###### grained recognition. arXiv preprint arXiv:2203.02751,

###### 2022. 12, 13

###### [87] Jihao Liu, Xin Huang, Yu Liu, and Hongsheng Li. Mixmim:

###### Mixed and masked image modeling for efficient visual

###### representation learning.arXiv preprint arXiv:2205.13137,

###### 2022. 12, 13


###### [88] Mannat Singh, Laura Gustafson, Aaron Adcock, Vinicius

###### de Freitas Reis, Bugra Gedik, Raj Prateek Kosaraju, Dhruv

###### Mahajan, Ross Girshick, Piotr Dollar, and Laurens Van ́

###### Der Maaten. Revisiting weakly supervised pre-training of

###### visual perception models. InIEEE Conf. Comput. Vis. Pat-

###### tern Recog., pages 804–814, 2022. 12, 13

###### [89] Huajun Liu, Fuqiang Liu, Xinyi Fan, and Dong Huang. Po-

###### larized self-attention: Towards high-quality pixel-wise re-

###### gression.arXiv preprint arXiv:2107.00782, 2021. 13

###### [90] Huayao Liu, Jiaming Zhang, Kailun Yang, Xinxin Hu, and

###### Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x

###### semantic segmentation with transformers. arXiv preprint

###### arXiv:2203.04838, 2022. 13, 14

###### [91] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-

###### Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang,

###### Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2:

###### Unifying localization and vision-language understanding.

###### arXiv preprint arXiv:2206.05836, 2022. 13

###### [92] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu,

###### Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue

###### Cao. Eva: Exploring the limits of masked visual represen-

###### tation learning at scale.arXiv preprint arXiv:2211.07636,

###### 2022. 13

###### [93] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian,

###### Tsung-Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph.

###### Simple copy-paste is a strong data augmentation method for

###### instance segmentation. InIEEE Conf. Comput. Vis. Pattern

###### Recog., pages 2918–2928, 2021. 13

###### [94] Xuan Jin, Wei Su, Rong Zhang, Yuan He, and Hui

###### Xue. Atldetv2. http://host.robots.ox.

###### ac.uk/leaderboard/displaylb_main.php?

###### challengeid=11&compid=4, 2019. 13

###### [95] Yu Liu, Guanglu Song, Yuhang Zang, Yan Gao, Enze Xie,

###### Junjie Yan, Chen Change Loy, and Xiaogang Wang. 1st

###### place solutions for openimage2019–object detection and

###### instance segmentation. arXiv preprint arXiv:2003.07557,

###### 2020. 13

###### [96] Anlin Zheng, Yuang Zhang, Xiangyu Zhang, Xiaojuan Qi,

###### and Jian Sun. Progressive end-to-end object detection

###### in crowded scenes. InIEEE Conf. Comput. Vis. Pattern

###### Recog., pages 857–866, 2022. 13

###### [97] Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang,

###### Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing

###### Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An

###### evolved version of yolo.arXiv preprint arXiv:2203.16250,

###### 2022. 13

###### [98] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui,

###### Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona,

###### and Serge Belongie. The inaturalist species classification

###### and detection dataset. InIEEE Conf. Comput. Vis. Pattern

###### Recog., pages 8769–8778, 2018. 12

###### [99] Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Tor-

###### ralba, and Aude Oliva. Learning deep features for scene

###### recognition using places database.Adv. Neural Inform. Pro-

###### cess. Syst., 27, 2014. 12

###### [100] Alejandro Lopez-Cifuentes, Marcos Escudero-Vinolo, ́

###### Jesus Besc ́ os, and ́ Alvaro Garc ́ ́ıa-Mart ́ın. Semantic-aware

###### scene recognition.Pattern Recognition, 102:107256, 2020.

###### 12

###### [101] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A

###### dataset for large vocabulary instance segmentation. InIEEE

###### Conf. Comput. Vis. Pattern Recog., pages 5356–5364, 2019.

###### 12

###### [102] Mark Everingham, Luc Van Gool, Christopher KI

###### Williams, John Winn, and Andrew Zisserman. The pascal

###### visual object classes (voc) challenge.Int. J. Comput. Vis.,

###### 88(2):303–338, 2010. 13

###### [103] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui-

###### jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Ste-

###### fan Popov, Matteo Malloci, Alexander Kolesnikov, et al.

###### The open images dataset v4. Int. J. Comput. Vis.,

###### 128(7):1956–1981, 2020. 13

###### [104] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu,

###### Xiangyu Zhang, and Jian Sun. Crowdhuman: A bench-

###### mark for detecting human in a crowd. arXiv preprint

###### arXiv:1805.00123, 2018. 13

###### [105] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingy-

###### ing Chen, Fangchen Liu, Vashisht Madhavan, and Trevor

###### Darrell. Bdd100k: A diverse driving dataset for hetero-

###### geneous multitask learning. InIEEE Conf. Comput. Vis.

###### Pattern Recog., pages 2636–2645, 2020. 13

###### [106] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu

###### Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and

###### Alan Yuille. The role of context for object detection and

###### semantic segmentation in the wild. InIEEE Conf. Comput.

###### Vis. Pattern Recog., pages 891–898, 2014. 13

###### [107] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo

###### Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe

###### Franke, Stefan Roth, and Bernt Schiele. The cityscapes

###### dataset for semantic urban scene understanding. InIEEE

###### Conf. Comput. Vis. Pattern Recog., 2016. 13

###### [108] Andrew Tao, Karan Sapra, and Bryan Catanzaro. Hierarchi-

###### cal multi-scale attention for semantic segmentation.arXiv

###### preprint arXiv:2005.10821, 2020. 14

###### [109] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and

###### Peter Kontschieder. The mapillary vistas dataset for seman-

###### tic understanding of street scenes. InInt. Conf. Comput.

###### Vis., pages 4990–4999, 2017. 14

###### [110] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob

###### Fergus. Indoor segmentation and support inference from

###### rgbd images.Eur. Conf. Comput. Vis., 7576:746–760, 2012.

###### 14


