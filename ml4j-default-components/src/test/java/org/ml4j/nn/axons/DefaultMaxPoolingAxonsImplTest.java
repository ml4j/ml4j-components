package org.ml4j.nn.axons;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.mockito.Mockito;

public class DefaultMaxPoolingAxonsImplTest extends Axons3DTestBase<MaxPoolingAxons> {

	@Before
	@Override
	public void setUp() {
		super.setUp();
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(400 * 2);
		Mockito.when(leftNeurons.getDepth()).thenReturn(2);
		Mockito.when(leftNeurons.getWidth()).thenReturn(20);
		Mockito.when(leftNeurons.getHeight()).thenReturn(20);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(100 * 2);
		Mockito.when(rightNeurons.getDepth()).thenReturn(2);
		Mockito.when(rightNeurons.getWidth()).thenReturn(10);
		Mockito.when(rightNeurons.getHeight()).thenReturn(10);
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockLeftToRightInputActivation = createNeuronsActivation(400 * 2, 32);
	}

	@Override
	protected MaxPoolingAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		return new DefaultMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config, false);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new ImageNeuronsActivationImpl(
				matrixFactory.createMatrix(featureCount, exampleCount),
				leftNeurons,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
	}

	@Override
	protected int getExpectedReformattedInputColumns() {
		return 32 * 100 * 2;
	}

	@Override
	protected int getExpectedReformattedInputRows() {
		return 11 * 11;
	}

	@Override
	protected boolean expectPostDropoutInputToBeSet() {
		return false;
	}

}
