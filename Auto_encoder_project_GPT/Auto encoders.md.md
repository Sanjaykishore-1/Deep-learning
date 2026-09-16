# Build a Production-Grade MNIST Denoising Autoencoder Web App

You are a senior full-stack engineer, ML engineer, UI/UX designer, and frontend architect.

Build a complete, production-quality web application around the supplied trained MNIST convolutional autoencoder.

I have provided two important project assets:

1. `Auto encoders.ipynb`
2. `autoencoder_model.h5`

You MUST inspect and use these files as the source of truth for the ML model rather than creating a different model from scratch.

The application must be deployable to Netlify and should feel like a polished AI product rather than a typical machine-learning demo.

---

# 1. Existing ML Model — DO NOT CHANGE THE MODEL LOGIC

The supplied notebook contains a convolutional denoising autoencoder trained using MNIST.

The relevant preprocessing is:

```python
X_Train = X_train / 255.0
X_Test = X_test / 255.0

X_train_noise = X_Train + 0.2 * np.random.normal(
    loc=0.0,
    scale=1,
    size=X_train.shape
)

X_test_noise = X_Test + 0.2 * np.random.normal(
    loc=0.0,
    scale=1,
    size=X_test.shape
)

X_train_clipped = np.clip(X_train_noise, 0, 1)
X_test_clipped = np.clip(X_test_noise, 0, 1)
```

The trained architecture is:

```python
model = Sequential()

# Encoder
model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same',
        input_shape=(28, 28, 1)
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        16,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(MaxPooling2D((2, 2)))

# Decoder
model.add(
    Conv2D(
        16,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(UpSampling2D((2, 2)))

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(UpSampling2D((2, 2)))

model.add(
    Conv2D(
        1,
        (3, 3),
        activation='sigmoid',
        padding='same'
    )
)
```

The model was compiled using:

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)
```

The model was trained to reconstruct the clean image from a noisy image:

```text
NOISY MNIST IMAGE
       ↓
     ENCODER
       ↓
 LATENT REPRESENTATION
       ↓
     DECODER
       ↓
RECONSTRUCTED CLEAN IMAGE
```

The original model was saved as:

```text
autoencoder_model.h5
```

Do NOT replace the architecture with a newly invented autoencoder.

---

# 2. PRIMARY TECHNICAL REQUIREMENT

The application must work with the supplied trained model.

Because the final application will be deployed on Netlify, prefer a client-side inference architecture.

Recommended approach:

```text
autoencoder_model.h5
        ↓
TensorFlow.js conversion
        ↓
model.json + weight shard files
        ↓
Netlify static deployment
        ↓
Browser loads model
        ↓
TensorFlow.js performs inference locally
```

The application should NOT require a continuously running Python/TensorFlow backend unless there is a technical reason that makes browser inference impossible.

If TensorFlow.js conversion is required, create/document the conversion process.

The final repository should contain the converted browser model in an appropriate directory such as:

```text
/public/model/
    model.json
    group1-shard1ofN.bin
    ...
```

Do not commit unnecessary duplicate model formats if they are not required for production.

---

# 3. IMPORTANT MODEL VALIDATION

Before building the UI, validate the supplied H5 model.

Confirm:

- input shape
- output shape
- number of layers
- activation functions
- preprocessing
- output range
- model loading compatibility
- inference result dimensions

Expected input:

```text
28 × 28 × 1
```

Expected output:

```text
28 × 28 × 1
```

Input values must ultimately be represented in:

```text
0.0 → 1.0
```

The output uses sigmoid and therefore should also be interpreted in the:

```text
0.0 → 1.0
```

range.

If TensorFlow.js conversion produces any compatibility issue, investigate and resolve it rather than silently substituting another model.

---

# 4. APPLICATION CONCEPT

Do NOT create a conventional ML dashboard.

Do NOT make it look like:

```text
Upload Image
[button]

Original Image | Result Image

Prediction: ...
```

That type of interface is too generic.

Instead, design the application as an interactive **AI Image Restoration Laboratory**.

The central visual metaphor should be:

> "Watch an AI reconstruct a damaged digit."

The experience should feel closer to an experimental digital laboratory / AI research interface than a basic image-upload website.

Possible product name:

# NEURAL RECONSTRUCT

Subtitle:

> See how an AI learns to recover what noise tries to erase.

You may improve the naming if you find something stronger.

---

# 5. VISUAL DESIGN DIRECTION

Create a highly polished, modern interface.

Visual direction:

- futuristic
- minimal
- premium
- experimental
- intelligent
- technical without being overwhelming
- elegant
- responsive
- visually memorable

Avoid generic SaaS dashboard aesthetics.

Avoid excessive cards.

Avoid excessive gradients.

Avoid stock illustrations.

Avoid unnecessary decorative AI graphics.

Use the actual digit/image reconstruction as the hero visual.

The application should have a dark experimental-lab aesthetic by default.

Possible design language:

- deep charcoal/black background
- subtle grid/noise texture
- soft glass surfaces
- thin borders
- restrained glow
- monochrome image visualizations
- one accent color
- subtle animated particles or scan lines
- high-quality typography

Do not make the interface look like a gaming website.

It should feel like an advanced AI research instrument.

---

# 6. HERO EXPERIENCE

The first viewport should immediately explain the application.

Hero section:

Small label:

```text
CONVOLUTIONAL AUTOENCODER / MNIST
```

Large heading:

```text
Watch noise
become a number.
```

Supporting text:

```text
A convolutional denoising autoencoder reconstructs handwritten digits
from corrupted visual input.
```

Then immediately show the interactive reconstruction laboratory.

Do not make users scroll through a large marketing page before using the model.

The actual AI interaction should be the primary experience.

---

# 7. MAIN INTERACTIVE LAB

Create a large central workspace.

Instead of simple side-by-side cards, create a visual pipeline:

```text
INPUT
  ↓
CORRUPTION
  ↓
ENCODER
  ↓
LATENT SPACE
  ↓
DECODER
  ↓
RECONSTRUCTION
```

The user should be able to visually understand what the model is doing.

Use an animated horizontal/vertical neural pipeline.

Example:

```text
┌──────────┐
│ ORIGINAL │
│   DIGIT  │
└────┬─────┘
     ↓
┌──────────┐
│  NOISE   │
│ INJECTION│
└────┬─────┘
     ↓
┌──────────┐
│ ENCODER  │
│ 32 → 16  │
└────┬─────┘
     ↓
┌──────────┐
│ LATENT   │
│ 7 × 7    │
└────┬─────┘
     ↓
┌──────────┐
│ DECODER  │
│ 16 → 32  │
└────┬─────┘
     ↓
┌──────────┐
│ CLEAN    │
│ OUTPUT   │
└──────────┘
```

The UI does not need to literally reproduce this ASCII design.

Create a beautiful visual equivalent.

---

# 8. DRAW YOUR OWN DIGIT

This should be the primary interaction.

Provide a large digital canvas approximately representing a 28×28 MNIST image.

Users should be able to draw a digit with their mouse or touch.

Requirements:

- mouse support
- touch support
- pointer events
- pressure-aware drawing if available
- smooth brush
- black/white or white-on-black visualization
- clear button
- undo if practical
- stroke smoothing
- automatic centering
- automatic resizing to 28×28
- normalization to 0–1

The canvas may display at a much larger size visually, but the actual model input must become:

```text
28 × 28 × 1
```

before inference.

---

# 9. AUTOMATIC PREPROCESSING

When a user draws an image:

1. Read canvas pixels.
2. Convert to grayscale.
3. Detect the relevant bounding box where possible.
4. Remove excessive empty margins.
5. Center the digit.
6. Resize appropriately.
7. Convert to 28×28.
8. Normalize pixel values to 0–1.
9. Add the channel dimension.
10. Run the autoencoder.

Do not distort the digit unnecessarily.

Keep preprocessing consistent and predictable.

---

# 10. NOISE CONTROL

Include an interactive noise control.

Example:

```text
NOISE LEVEL

○────────────●────────
0%          20%     50%
```

Default:

```text
20%
```

This default should correspond conceptually to the training notebook's:

```python
0.2 * np.random.normal(...)
```

The user should be able to increase/decrease noise and visually see its effect.

When noise changes, show:

```text
CLEAN INPUT
     ↓
NOISY INPUT
     ↓
RECONSTRUCTION
```

The UI should make the effect visually obvious.

---

# 11. RECONSTRUCTION ANIMATION

Do not instantly replace the image.

Create a short inference/reconstruction animation.

For example:

```text
Loading model...
      ↓
Encoding...
      ↓
Compressing...
      ↓
Latent representation...
      ↓
Decoding...
      ↓
Reconstructing...
```

Use this as a visual storytelling mechanism.

Do not fake model computation.

The animation should represent the actual inference process, but the timings can be UI transitions around the real inference call.

The user should feel that they are observing an AI reconstruction process.

---

# 12. IMAGE COMPARISON

After inference, show a sophisticated comparison view.

Include:

```text
INPUT
NOISY
RECONSTRUCTED
```

Potential interaction:

- draggable comparison slider
- before/after reveal
- pixel zoom
- image magnification
- toggle between views

Include a "Pixel View" mode where the 28×28 image is enlarged and individual pixels become visible.

This would make the application educational as well as visually impressive.

---

# 13. MODEL INSIGHT PANEL

Create a collapsible panel called:

```text
UNDER THE HOOD
```

It should explain the actual model.

Show:

```text
Input
28 × 28 × 1
```

```text
Encoder

Conv2D
32 filters

MaxPooling
14 × 14

Conv2D
16 filters

MaxPooling
7 × 7
```

```text
Latent representation
7 × 7 × 16
```

```text
Decoder

Conv2D
16 filters

UpSampling
14 × 14

Conv2D
32 filters

UpSampling
28 × 28

Conv2D
1 filter
Sigmoid
```

Do not invent parameter counts if they are not verified from the supplied model.

If parameter counts are displayed, calculate them programmatically or obtain them directly from the model.

---

# 14. LATENT SPACE VISUALIZATION

If technically feasible, provide a visual representation of the latent activation.

The model's bottleneck is:

```text
7 × 7 × 16
```

Show a stylized latent-space visualization.

For example:

```text
7 × 7 feature grid
```

with 16 channels.

The user could select a channel and see the activation map.

This feature is optional only if it can be implemented correctly from the actual model.

Do NOT fabricate latent values.

If browser-side extraction of intermediate layer activations is practical with TensorFlow.js, implement it.

Otherwise omit this feature rather than faking it.

---

# 15. PRESET EXAMPLES

Include several built-in examples so the user can immediately understand the application.

Example buttons:

```text
01
02
03
04
05
07
08
09
```

These can be generated from suitable MNIST examples or supplied static sample images.

Clicking a preset should populate the input area and run reconstruction.

Also include:

```text
Random Digit
```

which selects a random sample.

---

# 16. CUSTOM IMAGE UPLOAD

Allow users to upload an image.

Supported formats:

```text
PNG
JPG
JPEG
WEBP
```

However, explain that the model was trained specifically around MNIST-like grayscale handwritten digits.

After upload:

1. convert to grayscale
2. resize/crop
3. center
4. normalize
5. convert to 28×28×1
6. run reconstruction

Show a small warning if the uploaded image is very different from MNIST.

Example:

```text
MNIST-trained model
Best results occur with centered handwritten digits.
```

Do not pretend the model can reconstruct arbitrary photographs.

---

# 17. METRICS

Show useful reconstruction information.

Possible metrics:

```text
Input Noise
20%

Image Size
28 × 28

Reconstruction Error
0.XXXX

Inference Time
XX ms
```

The reconstruction error should be calculated from the actual input/target available in the application.

IMPORTANT:

For arbitrary user input there may be no clean ground-truth image.

Therefore do NOT calculate a fake reconstruction accuracy.

If no ground truth exists, clearly distinguish:

```text
Reconstruction similarity
```

from actual training/test accuracy.

Never display misleading "AI confidence".

---

# 18. RESPONSIVE DESIGN

The application must work beautifully on:

- desktop
- laptop
- tablet
- mobile

On desktop:

Use a spacious laboratory layout.

On mobile:

Stack the experience:

```text
Draw
↓
Noise
↓
Reconstruct
↓
Compare
↓
Model
```

The drawing canvas must remain comfortable to use on touch devices.

Do not create horizontal overflow.

---

# 19. ACCESSIBILITY

Implement:

- keyboard navigation
- visible focus states
- accessible buttons
- semantic HTML
- ARIA labels where appropriate
- sufficient contrast
- reduced-motion support
- screen-reader-friendly controls

Respect:

```text
prefers-reduced-motion
```

and disable nonessential animations when enabled.

---

# 20. TECHNOLOGY STACK

Preferred stack:

```text
React
TypeScript
Vite
Tailwind CSS
TensorFlow.js
```

Use a modern component architecture.

Do not introduce unnecessary frameworks.

Use TensorFlow.js for browser inference.

Use a suitable lightweight icon library if needed.

Avoid massive dependencies for simple functionality.

---

# 21. PROJECT STRUCTURE

Create a clean structure similar to:

```text
/
├── public/
│   ├── model/
│   │   ├── model.json
│   │   └── *.bin
│   └── samples/
│
├── src/
│   ├── components/
│   │   ├── DrawingCanvas.tsx
│   │   ├── NoiseControl.tsx
│   │   ├── ReconstructionView.tsx
│   │   ├── NeuralPipeline.tsx
│   │   ├── ModelInspector.tsx
│   │   ├── PixelViewer.tsx
│   │   ├── PresetGallery.tsx
│   │   └── Header.tsx
│   │
│   ├── lib/
│   │   ├── model.ts
│   │   ├── preprocessing.ts
│   │   ├── inference.ts
│   │   └── imageUtils.ts
│   │
│   ├── hooks/
│   │   └── useAutoencoder.ts
│   │
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
│
├── Auto encoders.ipynb
├── autoencoder_model.h5
├── package.json
├── vite.config.ts
├── netlify.toml
└── README.md
```

Adjust the structure when appropriate.

---

# 22. MODEL SERVICE

Create a clean abstraction such as:

```typescript
loadAutoencoder()
```

and:

```typescript
reconstruct(image)
```

The UI must not directly contain TensorFlow.js tensor manipulation everywhere.

Keep ML logic isolated.

Example conceptual flow:

```text
UI
 ↓
useAutoencoder()
 ↓
inference.ts
 ↓
TensorFlow.js
 ↓
model.json
```

---

# 23. TENSOR MANAGEMENT

Be extremely careful about TensorFlow.js memory management.

Dispose tensors appropriately.

Avoid memory leaks during repeated inference.

For example, repeated:

```text
draw → reconstruct → change noise → reconstruct → reconstruct...
```

must not continuously increase browser memory usage.

Use appropriate tensor scopes and cleanup.

---

# 24. MODEL LOADING UX

When the application opens:

```text
INITIALIZING NEURAL ENGINE
```

Show model loading progress if possible.

Example:

```text
Loading model
████████████░░░░ 78%
```

Then:

```text
MODEL READY
```

If model loading fails, provide a clear error state.

Do not leave the user staring at a blank screen.

---

# 25. OFFLINE/PRIVACY BENEFIT

Since inference is performed in the browser, communicate this clearly:

```text
LOCAL INFERENCE

Your drawing is processed in your browser.
No image needs to be uploaded to a server.
```

Only display this if the final architecture truly performs inference locally.

Do not claim privacy guarantees beyond what the implementation actually provides.

---

# 26. NETLIFY DEPLOYMENT

The application must be compatible with Netlify.

Configure:

```text
Build command:
npm run build
```

Publish directory:

```text
dist
```

Create an appropriate `netlify.toml`.

Ensure TensorFlow.js model files are correctly included in the final production build.

The deployed application must be able to resolve:

```text
/model/model.json
```

and all weight shard files.

Do not use local filesystem paths.

Do not reference:

```text
C:\...
/Users/...
```

or notebook paths.

---

# 27. NETLIFY CONFIGURATION

Create appropriate SPA fallback configuration so React routes do not return 404.

For example, configure Netlify to serve the application entry point for client-side routes.

Do not break static model asset requests.

Verify that:

```text
/model/model.json
```

continues to resolve correctly.

---

# 28. ENVIRONMENT VARIABLES

The application should ideally require no secret environment variables because inference happens client-side.

If any environment variable is genuinely required, document it clearly.

Never expose private API keys in frontend source code.

---

# 29. PERFORMANCE

Optimize for production.

Requirements:

- lazy-load optional components
- avoid unnecessary rerenders
- avoid recreating the model on every inference
- load the model once
- reuse the loaded model
- dispose tensors correctly
- compress/static-serve model assets
- keep animations lightweight
- avoid excessive GPU/CPU work

The app should feel responsive on normal laptops and modern mobile devices.

---

# 30. ERROR HANDLING

Handle:

- model loading failure
- unsupported browser
- TensorFlow.js initialization failure
- invalid image
- canvas errors
- inference failure
- oversized upload
- malformed model files

Provide human-readable errors.

Example:

```text
The neural model could not be loaded.
Please refresh the page and try again.
```

Do not expose raw stack traces to normal users.

---

# 31. DEMO MODE / FALLBACK

Do NOT silently replace the actual model with a fake reconstruction.

If model loading fails, provide an explicit development/debug state.

For production, the application should fail clearly rather than pretending inference succeeded.

---

# 32. VISUAL MICRO-INTERACTIONS

Use subtle animations:

- neural pathway pulse
- model loading
- image reconstruction
- scan line
- panel expansion
- slider movement
- button hover
- canvas interaction
- latent activation transitions

Animations should be restrained.

The goal is:

```text
premium AI laboratory
```

not:

```text
gaming UI
```

---

# 33. TYPOGRAPHY

Use a modern technical typeface pairing.

For example:

- strong modern sans-serif for headings
- clean readable sans-serif for body
- monospace for technical model values

Technical labels can use uppercase tracking:

```text
MODEL STATUS
LATENT SPACE
INFERENCE
RECONSTRUCTION
```

Do not overuse uppercase for normal explanatory text.

---

# 34. LANDING PAGE INFORMATION ARCHITECTURE

The application should approximately follow:

```text
HEADER

HERO
"Watch noise become a number."

        ↓

INTERACTIVE RECONSTRUCTION LAB

        ↓

INPUT → NOISE → ENCODE → LATENT → DECODE → OUTPUT

        ↓

UNDER THE HOOD

        ↓

MODEL ARCHITECTURE

        ↓

HOW IT WORKS

        ↓

FOOTER
```

The interaction should remain accessible near the top.

---

# 35. "HOW IT WORKS" SECTION

Explain denoising autoencoders in simple language.

Use three stages:

### 01 — Corrupt

Noise is added to the digit.

### 02 — Compress

The encoder compresses visual information into a smaller representation.

### 03 — Reconstruct

The decoder uses that representation to reconstruct the clean digit.

Keep the explanation understandable to someone who knows basic AI but is not an expert.

---

# 36. EDUCATIONAL MODEL DETAILS

Include a technical section showing:

```text
Framework
TensorFlow / Keras

Architecture
Convolutional Autoencoder

Dataset
MNIST

Input
28 × 28 × 1

Task
Image Denoising

Loss
Mean Squared Error

Optimizer
Adam

Training Epochs
5
```

These values must reflect the supplied notebook.

Do not invent training statistics.

---

# 37. DO NOT MISREPRESENT THE MODEL

This is extremely important.

Do NOT claim:

- the model recognizes digits
- the model predicts digit classes
- the model has classification accuracy
- the model is a generative AI model
- the model was trained for arbitrary images
- the model understands handwriting semantically

It is a:

```text
convolutional denoising autoencoder
```

Its purpose is reconstruction/denoising.

---

# 38. OPTIONAL ADVANCED FEATURE — RECONSTRUCTION PLAYGROUND

If practical, create a section called:

```text
RECONSTRUCTION PLAYGROUND
```

Controls:

```text
Noise
[slider]

Brush Size
[slider]

Contrast
[slider]

Auto Center
[toggle]

Pixel Grid
[toggle]
```

The user can experiment and observe how preprocessing affects reconstruction.

---

# 39. OPTIONAL ADVANCED FEATURE — PIXEL GRID

Provide a toggle:

```text
PIXEL GRID
```

When enabled, display the 28×28 pixel boundaries.

Hovering over a pixel can display something like:

```text
x: 14
y: 18
value: 0.72
```

This should make the application feel like an actual ML visualization tool.

---

# 40. SOURCE CODE QUALITY

Write clean TypeScript.

Avoid:

```text
any
```

where it can reasonably be avoided.

Use reusable components.

Use meaningful variable names.

Separate:

```text
presentation
state
ML inference
image processing
```

Do not create a single giant `App.tsx`.

---

# 41. README

Create a comprehensive README containing:

1. Project overview
2. Model architecture
3. Existing notebook explanation
4. H5 model information
5. TensorFlow.js conversion instructions
6. Local development
7. Build command
8. Netlify deployment
9. Model file location
10. Troubleshooting
11. Architecture diagram
12. Limitations

Include commands where required.

For example, document how the supplied H5 model was converted to TensorFlow.js.

Do not assume the user already knows the conversion process.

---

# 42. MODEL CONVERSION

If conversion from:

```text
autoencoder_model.h5
```

to TensorFlow.js is necessary, create a repeatable conversion workflow.

The conversion should produce:

```text
model.json
*.bin
```

Validate the converted model against the original H5 model.

Use the same test image for both models.

Compare:

```text
H5 prediction
vs
TensorFlow.js prediction
```

The results should be sufficiently close within normal floating-point tolerance.

Document the validation.

---

# 43. TESTING

Create tests for:

### Preprocessing

Verify:

```text
arbitrary image
→ grayscale
→ centered
→ 28×28
→ normalized
→ 28×28×1
```

### Model

Verify:

```text
input shape = 28×28×1
output shape = 28×28×1
```

### UI

Verify:

- drawing works
- clear works
- noise slider works
- preset works
- upload works
- reconstruction works
- model loading state works
- error state works
- mobile layout works

### Deployment

Verify:

```text
npm run build
```

and verify that the production build can load the TensorFlow.js model.

---

# 44. FINAL QA REQUIREMENT

Before considering the project complete, perform an end-to-end test:

```text
Open application
      ↓
Model loads
      ↓
Draw digit
      ↓
Apply noise
      ↓
Run reconstruction
      ↓
Display reconstructed image
      ↓
Change noise
      ↓
Run reconstruction again
      ↓
Upload image
      ↓
Run reconstruction
      ↓
Check mobile layout
      ↓
Production build
      ↓
Netlify-compatible output
```

Fix all console errors.

There should be no:

```text
404 model errors
CORS errors
TensorFlow tensor leaks
broken asset paths
React warnings
TypeScript build errors
```

---

# 45. IMPORTANT DESIGN PRINCIPLE

The application should make the user think:

> "I'm interacting with the inside of an AI model."

Not:

> "I'm looking at a webpage that has an AI model attached to it."

The neural pipeline, image transformation, latent representation, pixel visualization, and reconstruction animation should all contribute to this feeling.

---

# 46. FINAL DELIVERABLE

Produce the complete project.

I expect:

```text
Frontend
+
TensorFlow.js model integration
+
Image preprocessing
+
Drawing interface
+
Noise simulation
+
Reconstruction visualization
+
Model architecture visualization
+
Responsive UI
+
Accessibility
+
Error handling
+
Testing
+
Netlify configuration
+
README
```

Do not provide only a mockup.

Do not use placeholder model inference.

Do not create fake reconstruction results.

Use the supplied `autoencoder_model.h5`.

Use the supplied `Auto encoders.ipynb` to validate preprocessing and architecture.

The final application must be genuinely connected to the trained model.

---

# 47. MOST IMPORTANT IMPLEMENTATION RULE

Before writing the UI, inspect the supplied:

```text
Auto encoders.ipynb
autoencoder_model.h5
```

and verify that the implementation matches the actual trained model.

If anything in this prompt conflicts with the actual model files, prioritize the actual model files and explain the discrepancy in the README.

Do not silently modify the ML architecture.

Build the application as a polished, production-ready Netlify-deployable AI experience.