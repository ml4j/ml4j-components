package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

public class DummyAveragePoolingAxonsImplTest extends Axons3DTestBase<AveragePoolingAxons> {

	private MatrixFactory matrixFactory;

	@Before
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
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
	protected AveragePoolingAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config) {
		return new DummyAveragePoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
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
		return 400 * 2;
	}

	@Override
	protected boolean expectPostDropoutInputToBeSet() {
		return true;
	}

}
