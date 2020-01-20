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
package org.ml4j.nn.components.manytoone;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

/**
 * Unit test for DefaultManyToOneFilterConcatDirectedComponentImpl.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultManyToOneFilterConcatDirectedComponentImplTest extends ManyToOneDirectedComponentTestBase {

	@Override
	protected ManyToOneDirectedComponent<?> createManyToOneDirectedComponentUnderTest(
			PathCombinationStrategy pathCombinationStrategy, Neurons outputNeurons) {
		if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT && outputNeurons instanceof Neurons3D) {
			return new DefaultManyToOneFilterConcatDirectedComponentImpl((Neurons3D) outputNeurons);
		} else {
			throw new IllegalArgumentException(
					"Path combination strategy must be FILTER_CONCAT and output neurons must be 3D neurons for this test");
		}
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		throw new UnsupportedOperationException();
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

}
