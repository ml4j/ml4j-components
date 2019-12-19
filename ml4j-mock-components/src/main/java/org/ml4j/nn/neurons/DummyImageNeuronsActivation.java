package org.ml4j.nn.neurons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;

public class DummyImageNeuronsActivation extends DummyNeuronsActivation implements ImageNeuronsActivation {

	public DummyImageNeuronsActivation(Neurons3D neurons, NeuronsActivationFeatureOrientation featureOrientation, int examples) {
		super(neurons, featureOrientation, examples);
	}

	@Override
	public Images getImages() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Neurons3D getNeurons() {
		return (Neurons3D)neurons;
	}

	@Override
	public Matrix im2Col(MatrixFactory arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6) {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix im2Col2(MatrixFactory arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DummyImageNeuronsActivation dup() {
		return new DummyImageNeuronsActivation(getNeurons(), featureOrientation, examples);
	}
	
	

}
