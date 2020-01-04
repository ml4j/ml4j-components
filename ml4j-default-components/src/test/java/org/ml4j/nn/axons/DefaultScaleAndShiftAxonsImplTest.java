package org.ml4j.nn.axons;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.AxonsTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultScaleAndShiftAxonsImplTest extends AxonsTestBase<ScaleAndShiftAxons<?>> {


	@Before
	@Override
	public void setUp() {
		super.setUp();
	}

	@Override
	protected ScaleAndShiftAxons<?> createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		return new DefaultScaleAndShiftAxonsImpl<>(matrixFactory, leftNeurons, rightNeurons);
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
