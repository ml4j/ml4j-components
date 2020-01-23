/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.neurons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;

/**
 * Mock implementation of ImageNeuronsActivation
 * 
 * @author Michael Lavelle
 */
public class DummyImageNeuronsActivation extends DummyNeuronsActivation implements ImageNeuronsActivation {

	/**
	 * @param neurons            The neurons that have been activated.
	 * @param featureOrientation The feature orientation of the matrix
	 *                           representation of these activations.
	 * @param examples           The number of training examples that caused the
	 *                           activations.
	 */
	public DummyImageNeuronsActivation(Neurons3D neurons, NeuronsActivationFeatureOrientation featureOrientation,
			int examples) {
		super(neurons, featureOrientation, examples);
	}

	@Override
	public Images getImages() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Neurons3D getNeurons() {
		return (Neurons3D) neurons;
	}

	@Override
	public Matrix im2ColConv(MatrixFactory arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6) {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix im2ColPool(MatrixFactory arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DummyImageNeuronsActivation dup() {
		return new DummyImageNeuronsActivation(getNeurons(), featureOrientation, examples);
	}
	
	@Override
	public ImageNeuronsActivationFormat getFormat() {
		return ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT;
	}
}
