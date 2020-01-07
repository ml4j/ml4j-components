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
package org.ml4j.nn.components.manytomany;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.manytomany.base.DefaultDirectedComponentChainBatchTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mockito;

/**
 * Unit test for DefaultComponentChainBatchImpl.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultComponentChainBatchImplTest extends DefaultDirectedComponentChainBatchTestBase {

	@Override
	protected DefaultDirectedComponentChainBatch createDefaultDirectedComponentChainBatchUnderTest(
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		
		List<DefaultDirectedComponentChain> chains = components.stream().map(c -> new DefaultDirectedComponentChainImpl(Arrays.asList(c))).collect(Collectors.toList());
		return new DefaultComponentChainBatchImpl(chains);
	}
	
	@Override
	protected MatrixFactory createMatrixFactory() {
		return Mockito.mock(MatrixFactory.class);
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

}
