package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyConvolutionalAxonsImplTest extends Axons3DTestBase<ConvolutionalAxons> {

	@Before
	@Override
	public void setUp() {
		super.setUp();
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(784 * 3);
		Mockito.when(leftNeurons.getDepth()).thenReturn(3);
		Mockito.when(leftNeurons.getWidth()).thenReturn(28);
		Mockito.when(leftNeurons.getHeight()).thenReturn(28);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(400 * 2);
		Mockito.when(rightNeurons.getDepth()).thenReturn(2);
		Mockito.when(rightNeurons.getWidth()).thenReturn(20);
		Mockito.when(rightNeurons.getHeight()).thenReturn(20);
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockLeftToRightInputActivation = createNeuronsActivation(784 * 3, 32);
	}

	@Override
	protected ConvolutionalAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		return new DummyConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

	@Override
	protected int getExpectedReformattedInputColumns() {
		return 32;
	}

	@Override
	protected int getExpectedReformattedInputRows() {
		return 784 * 3;
	}

}
