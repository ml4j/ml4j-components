package org.ml4j.nn.axons.mocks;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;

public class DummyFullyConnectedAxonsImplTest extends AxonsTestBase<FullyConnectedAxons> {

	private MatrixFactory matrixFactory;

	@Before
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
		super.setUp();
	}

	@Override
	protected FullyConnectedAxons createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}

}
