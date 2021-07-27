import turicreate as tc
import coremltools
# Load the style and content images
styles = tc.load_images('./datasets/style/style/')
content = tc.load_images('./datasets/style/content/')

# Create a StyleTransfer model
model = tc.style_transfer.create(styles, content, max_iterations=2)

# Load some test images
test_images = tc.load_images('./datasets/style/test/')

# Stylize the test images
stylized_images = model.stylize(test_images)
print(stylized_images)
print(type(stylized_images))

# Save the model for later use in Turi Create
model.save('style-transfer.model')

# Export for use in Core ML
model.export_coreml('MyCustomStyleTransfer.mlmodel')