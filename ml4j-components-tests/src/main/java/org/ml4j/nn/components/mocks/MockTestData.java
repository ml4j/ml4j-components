package org.ml4j.nn.components.mocks;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.FeaturesFormat;
import org.mockito.Mockito;

public class MockTestData {

	public static NeuronsActivation mockNeuronsActivation(int featureCount, int exampleCount,
			MatrixFactory matrixFactory) {
		NeuronsActivation neuronsActivation = Mockito.mock(NeuronsActivation.class);
		Mockito.when(neuronsActivation.getFeatureCount()).thenReturn(featureCount);
		Mockito.when(neuronsActivation.getRows()).thenReturn(featureCount);
		Mockito.when(neuronsActivation.getColumns()).thenReturn(exampleCount);
		Mockito.when(neuronsActivation.getNeurons()).thenReturn(new Neurons(featureCount, false));
		Mockito.when(neuronsActivation.getFormat()).thenReturn(createMockFormat());
		Mockito.when(neuronsActivation.getExampleCount()).thenReturn(exampleCount);
		Mockito.when(neuronsActivation.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		Matrix mockMatrix = matrixFactory.createMatrix(featureCount, exampleCount);
		Mockito.when(neuronsActivation.getActivations(Mockito.any())).thenReturn(mockMatrix);
		Mockito.when(neuronsActivation.dup()).thenReturn(neuronsActivation);
		return neuronsActivation;
	}

	public static NeuronsActivation mockNeuronsActivation(int featureCount, int exampleCount) {
		Matrix mockMatrix = Mockito.mock(Matrix.class);
		Mockito.when(mockMatrix.getRows()).thenReturn(featureCount);
		Mockito.when(mockMatrix.getColumns()).thenReturn(exampleCount);
		return mockNeuronsActivation(featureCount, 1, 1, featureCount, exampleCount, mockMatrix);
	}

	public static NeuronsActivation mockNeuronsActivationForImage(int featureCount, int width, int height, int depth,
			int exampleCount) {
		Matrix mockMatrix = Mockito.mock(Matrix.class);
		Mockito.when(mockMatrix.getRows()).thenReturn(featureCount);
		Mockito.when(mockMatrix.getColumns()).thenReturn(exampleCount);
		return mockNeuronsActivation(featureCount, width, height, depth, exampleCount, mockMatrix);
	}

	public static NeuronsActivation mockNeuronsActivation(int featureCount, int width, int height, int depth,
			int exampleCount, Matrix matrix) {
		NeuronsActivation neuronsActivation = Mockito.mock(NeuronsActivation.class);
		Images mockImages = Mockito.mock(Images.class);
		Mockito.when(mockImages.getChannels()).thenReturn(depth);
		Mockito.when(mockImages.getWidth()).thenReturn(width);
		Mockito.when(mockImages.getHeight()).thenReturn(height);
		Mockito.when(mockImages.getExamples()).thenReturn(exampleCount);
		ImageNeuronsActivation mockImageActivation = Mockito.mock(ImageNeuronsActivation.class);
		Mockito.when(neuronsActivation.asImageNeuronsActivation(Mockito.any(), Mockito.any())).thenReturn(mockImageActivation);
		Mockito.when(mockImageActivation.getFeatureCount()).thenReturn(featureCount);
		Mockito.when(mockImageActivation.getExampleCount()).thenReturn(exampleCount);
		Mockito.when(mockImageActivation.getImages()).thenReturn(mockImages);

		Mockito.when(neuronsActivation.getFeatureCount()).thenReturn(featureCount);
		Mockito.when(neuronsActivation.getRows()).thenReturn(featureCount);
		Mockito.when(neuronsActivation.getColumns()).thenReturn(exampleCount);
		Mockito.when(neuronsActivation.getNeurons()).thenReturn(new Neurons(featureCount, false));
		Mockito.when(neuronsActivation.getExampleCount()).thenReturn(exampleCount);
		Mockito.when(neuronsActivation.getFeatureOrientation())
				.thenReturn(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		Mockito.when(neuronsActivation.getFormat()).thenReturn(createMockFormat());
		Mockito.when(mockImageActivation.getFormat()).thenReturn(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);

		Mockito.when(neuronsActivation.getActivations(Mockito.any())).thenReturn(matrix);
		Mockito.when(neuronsActivation.dup()).thenReturn(neuronsActivation);
		return neuronsActivation;
	}
	
	@SuppressWarnings("unchecked")
	private static <T extends FeaturesFormat> NeuronsActivationFormat<T> createMockFormat() {
		return (NeuronsActivationFormat<T>) NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET;
	}

	public static DirectedComponentGradient<NeuronsActivation> mockComponentGradient(int featureCount, int exampleCount,
			TestBase testInstance) {

		@SuppressWarnings("unchecked")
		DirectedComponentGradient<NeuronsActivation> gradient = Mockito.mock(DirectedComponentGradient.class);
		NeuronsActivation mockNeuronsActivation = testInstance.createNeuronsActivation(featureCount, exampleCount);
		Mockito.when(gradient.getOutput()).thenReturn(mockNeuronsActivation);
		return gradient;
	}

	public static DirectedComponentGradient<List<NeuronsActivation>> mockBatchComponentGradient(int featureCount,
			int exampleCount, int batchSize) {

		@SuppressWarnings("unchecked")
		DirectedComponentGradient<List<NeuronsActivation>> gradient = Mockito.mock(DirectedComponentGradient.class);
		List<NeuronsActivation> outputs = new ArrayList<>();
		for (int i = 0; i < batchSize; i++) {
			NeuronsActivation mockNeuronsActivation = mockNeuronsActivation(featureCount, exampleCount);
			outputs.add(mockNeuronsActivation);
		}
		Mockito.when(gradient.getOutput()).thenReturn(outputs);
		return gradient;
	}

	public static DirectedComponentGradient<List<NeuronsActivation>> mockBatchComponentGradient(int featureCount,
			int exampleCount, int batchSize, MatrixFactory matrixFactory) {

		@SuppressWarnings("unchecked")
		DirectedComponentGradient<List<NeuronsActivation>> gradient = Mockito.mock(DirectedComponentGradient.class);
		List<NeuronsActivation> outputs = new ArrayList<>();
		for (int i = 0; i < batchSize; i++) {
			NeuronsActivation mockNeuronsActivation = mockNeuronsActivation(featureCount, exampleCount, matrixFactory);
			outputs.add(mockNeuronsActivation);
		}
		Mockito.when(gradient.getOutput()).thenReturn(outputs);
		return gradient;
	}

	@SuppressWarnings("unchecked")
	public static DefaultChainableDirectedComponentActivation mockComponentActivation(int inputFeatureCount,
			int outputFeatureCount, int exampleCount, TestBase testInstance) {
		DefaultChainableDirectedComponentActivation mockActivation = Mockito
				.mock(DefaultChainableDirectedComponentActivation.class);
		NeuronsActivation mockOutput = mockNeuronsActivation(outputFeatureCount, exampleCount);
		Mockito.when(mockActivation.getOutput()).thenReturn(mockOutput);
		@SuppressWarnings("rawtypes")
		List decomposed = new ArrayList<>();
		decomposed.add(mockActivation);
		Mockito.when(mockActivation.decompose()).thenReturn(decomposed);
		DirectedComponentGradient<NeuronsActivation> mockGradient = mockComponentGradient(inputFeatureCount,
				exampleCount, testInstance);
		Mockito.when(mockActivation.backPropagate(Mockito.any())).thenReturn(mockGradient);
		return mockActivation;
	}
}
