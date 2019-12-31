package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;

public class DummyScaleAndShiftAxonsImplTest extends AxonsTestBase<ScaleAndShiftAxons<?>> {

	private MatrixFactory matrixFactory;

	@Before
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
		super.setUp();
	}

	@Override
	protected ScaleAndShiftAxons<?> createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		return new DummyScaleAndShiftAxonsImpl<>(matrixFactory, leftNeurons, rightNeurons);
	}

}
