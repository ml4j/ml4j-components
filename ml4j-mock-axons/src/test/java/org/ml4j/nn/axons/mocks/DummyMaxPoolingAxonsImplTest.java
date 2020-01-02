package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyMaxPoolingAxonsImplTest extends Axons3DTestBase<MaxPoolingAxons> {

	private MatrixFactory matrixFactory;

	@Before
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
		super.setUp();
	}

	@Override
	protected MaxPoolingAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config) {
		return new DummyMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
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
