

### First Test 02/04/25
First test began after first full implementation of DCT DWT.

Features were extracted using vgg19, dumping all its layers
[GitHub Repository for Neural Style Transfer](https://github.com/Magicmaan/Neural-Style-Transfer)

## Parameters
* DCT Alpha - 0.001
* DWT Alpha - [0.0001, 0.004, 0.004, 0.004]

# Results
* MSE Difference:  tensor(0.0365)
* Perceptual Difference:  0.040790580213069916
* Structural Difference:  tensor(0.9745)
* Peak Noise:  90.4090805053711

# Initial impressions
Splitting the DWT Alpha into its layers provided mildly better results but gave more tempermental results.
Changing the low alpha values seemed to generate more distortion, but more visibility
Changing the higher alpha values had less of an affect on the result.

The overall detection is low, with fairly low statistical results


# Images
<div style="display: flex; flex-direction: row; justify-content: space-around; align-items: center; margin-bottom: 1rem;">
    <div style="text-align: center;">
        <img src="./assets/lena.png" alt="Lena" style="width: 45%;"/>
        <p>Input</p>
    </div>
    <div style="text-align: center;">
        <img src="./assets/NST_TEST_01_watermarked.png" alt="NST Test 01 Watermarked" style="width: 45%;"/>
        <p>Output</p>
    </div>
</div>


![Process](./assets/NST_TEST_01_process.png)
The stages of copyright embedding

![Features](./assets/lena_features.png)
### Features Extracted from the Watermarked Image
Zoom in to observe the following details:

1. **Mild Pattern Retention**:  
    - Notice the subtle patterns retained in the second row (layers 4–7).

2. **Distortion in Lower Layers**:  
    - Observe the distortions generated in some of the lower layers.

These features highlight the impact of the watermarking process on the image's structure.



