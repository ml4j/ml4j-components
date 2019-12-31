package org.ml4j.nn.components.base;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class TestBase {

	protected MatrixFactory matrixFactory;
	
	public TestBase() {
		this.matrixFactory = createMatrixFactory();
	}

	protected abstract MatrixFactory createMatrixFactory();
	
	
	public abstract NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount);
	
}
