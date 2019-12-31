package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyAveragePoolingAxonsImplTest extends Axons3DTestBase<AveragePoolingAxons> {

	private MatrixFactory matrixFactory;

	@Before
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
		super.setUp();
	}

	@Override
	protected AveragePoolingAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth,
			int strideHeight, Integer paddingWidth, Integer paddingHeight) {
		return new DummyAveragePoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, strideWidth, strideHeight);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

}
