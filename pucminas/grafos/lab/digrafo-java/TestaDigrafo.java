public class TestaDigrafo{
	
	public static String stringCorreto(int valor, int esperado){
		return (esperado==valor)?"OK":"ERRADO";
	}

	public static String stringCorreto(boolean valor, boolean esperado){
		return (esperado==valor)?"OK":"ERRADO";
	}

	public static void verifica(String texto, int valor, int esperado){
		System.out.printf("%s: %d [%s]\n", texto, valor, stringCorreto(valor,esperado) );
	}

	public static void verifica(String texto, boolean valor, boolean esperado){
		System.out.printf("%s: %b [%s]\n", texto, valor, stringCorreto(valor,esperado) );
	}

	
	public static void testaMatrizDigrafo(){
		System.out.println("[Testando MatrizDigrafo]");
		Digrafo digrafo = new MatrizDigrafo(6);
		digrafo.adicionaAresta(0,1);
		digrafo.adicionaAresta(0,3);
		digrafo.adicionaAresta(1,4);
		digrafo.adicionaAresta(2,4);
		digrafo.adicionaAresta(2,5);
		digrafo.adicionaAresta(3,1);
		digrafo.adicionaAresta(4,3);
		digrafo.adicionaAresta(5,5);

		verifica("Numero vertices", digrafo.numVertices(), 6);
		verifica("Numero arestas", digrafo.numArestas(), 8);

		verifica("Existe aresta 0->1", digrafo.existeAresta(0,1), true);
		verifica("Existe aresta 0->3", digrafo.existeAresta(0,3), true);
		verifica("Existe aresta 1->4", digrafo.existeAresta(1,4), true);
		verifica("Existe aresta 2->4", digrafo.existeAresta(2,4), true);
		verifica("Existe aresta 2->5", digrafo.existeAresta(2,5), true);
		verifica("Existe aresta 3->1", digrafo.existeAresta(3,1), true);
		verifica("Existe aresta 4->3", digrafo.existeAresta(4,3), true);
		verifica("Existe aresta 5->5", digrafo.existeAresta(5,5), true);

		int arestasExistentes = 0;
		for(int u = 0; u<digrafo.numVertices(); u++){
			for(int v = 0; v<digrafo.numVertices(); v++){
				if(digrafo.existeAresta(u,v)){
					arestasExistentes++;
				}
			}
		}
		verifica("Arestas existentes",arestasExistentes,8);

		verifica("Grau de saida vertice do 0", digrafo.grauSaida(0), 2);
		verifica("Grau de saida vertice do 1", digrafo.grauSaida(1), 1);
		verifica("Grau de saida vertice do 2", digrafo.grauSaida(2), 2);
		verifica("Grau de saida vertice do 3", digrafo.grauSaida(3), 1);
		verifica("Grau de saida vertice do 4", digrafo.grauSaida(4), 1);
		verifica("Grau de saida vertice do 5", digrafo.grauSaida(5), 1);

		verifica("Grau de entrada vertice do 0", digrafo.grauEntrada(0), 0);
		verifica("Grau de entrada vertice do 1", digrafo.grauEntrada(1), 2);
		verifica("Grau de entrada vertice do 2", digrafo.grauEntrada(2), 0);
		verifica("Grau de entrada vertice do 3", digrafo.grauEntrada(3), 2);
		verifica("Grau de entrada vertice do 4", digrafo.grauEntrada(4), 2);
		verifica("Grau de entrada vertice do 5", digrafo.grauEntrada(5), 2);

		verifica("Existe loop", digrafo.existeLoop(), true);

		System.out.println("[Concluido MatrizDigrafo]");
		System.out.println();
	}

	public static void testaListaDigrafo(){
		System.out.println("[Testando ListaDigrafo]");
		Digrafo digrafo = new ListaDigrafo(6);
		digrafo.adicionaAresta(0,1);
		digrafo.adicionaAresta(0,3);
		digrafo.adicionaAresta(1,4);
		digrafo.adicionaAresta(2,4);
		digrafo.adicionaAresta(2,5);
		digrafo.adicionaAresta(3,1);
		digrafo.adicionaAresta(4,3);
		digrafo.adicionaAresta(5,5);

		verifica("Numero vertices", digrafo.numVertices(), 6);
		verifica("Numero arestas", digrafo.numArestas(), 8);

		verifica("Existe aresta 0->1", digrafo.existeAresta(0,1), true);
		verifica("Existe aresta 0->3", digrafo.existeAresta(0,3), true);
		verifica("Existe aresta 1->4", digrafo.existeAresta(1,4), true);
		verifica("Existe aresta 2->4", digrafo.existeAresta(2,4), true);
		verifica("Existe aresta 2->5", digrafo.existeAresta(2,5), true);
		verifica("Existe aresta 3->1", digrafo.existeAresta(3,1), true);
		verifica("Existe aresta 4->3", digrafo.existeAresta(4,3), true);
		verifica("Existe aresta 5->5", digrafo.existeAresta(5,5), true);

		int arestasExistentes = 0;
		for(int u = 0; u<digrafo.numVertices(); u++){
			for(int v = 0; v<digrafo.numVertices(); v++){
				if(digrafo.existeAresta(u,v)){
					arestasExistentes++;
				}
			}
		}
		verifica("Arestas existentes",arestasExistentes,8);

		verifica("Grau de saida vertice do 0", digrafo.grauSaida(0), 2);
		verifica("Grau de saida vertice do 1", digrafo.grauSaida(1), 1);
		verifica("Grau de saida vertice do 2", digrafo.grauSaida(2), 2);
		verifica("Grau de saida vertice do 3", digrafo.grauSaida(3), 1);
		verifica("Grau de saida vertice do 4", digrafo.grauSaida(4), 1);
		verifica("Grau de saida vertice do 5", digrafo.grauSaida(5), 1);

		verifica("Grau de entrada vertice do 0", digrafo.grauEntrada(0), 0);
		verifica("Grau de entrada vertice do 1", digrafo.grauEntrada(1), 2);
		verifica("Grau de entrada vertice do 2", digrafo.grauEntrada(2), 0);
		verifica("Grau de entrada vertice do 3", digrafo.grauEntrada(3), 2);
		verifica("Grau de entrada vertice do 4", digrafo.grauEntrada(4), 2);
		verifica("Grau de entrada vertice do 5", digrafo.grauEntrada(5), 2);

		verifica("Existe loop", digrafo.existeLoop(), true);

		System.out.println("[Concluido ListaDigrafo]");
		System.out.println();
	}

	public static void main(String []args){
		testaMatrizDigrafo();
		testaListaDigrafo();
	}
}
