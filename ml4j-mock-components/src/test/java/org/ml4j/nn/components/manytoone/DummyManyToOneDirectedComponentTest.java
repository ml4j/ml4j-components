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
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

/**
 * Unit test for DummyManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public class DummyManyToOneDirectedComponentTest extends ManyToOneDirectedComponentTestBase {

	@Override
	protected ManyToOneDirectedComponent<?> createManyToOneDirectedComponentUnderTest(
			PathCombinationStrategy pathCombinationStrategy, Neurons outputNeurons) {
		return new DummyManyToOneDirectedComponent(pathCombinationStrategy);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

}
