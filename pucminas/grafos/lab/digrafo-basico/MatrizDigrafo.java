/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos;
 *         Matriz de adjacencias.
 */

/**
 * Classe que implementa um grafo direcionado, ou digrafo.
 */
public class MatrizDigrafo extends Digrafo {
	private int n;
	private int [][]adj;

   /**
    * Construtor de um digrafo de n vertices, sem nenhuma aresta.
    * As arestas devem ser adicionadas posteriormente.
	 * Os vertices sao rotulados de 0 ate n-1.
    */
	public MatrizDigrafo(int n){
		this.n = n;
		this.adj = new int[n][n];
	}

   /**
    * Adiciona uma nova aresta direcionada do vertice u para v.
    */
	public void adicionaAresta(int u, int v){
		this.adj[u][v] = 1;
	}

   /**
    * Remove uma nova aresta direcionada do vertice u para v, caso a mesma exista.
    */
	public void removeAresta(int u, int v){
		this.adj[u][v] = 0;
	}

   /**
    * Verifica se existe uma aresta direcionada do vertice u para v.
    * @return true caso existir a aresta (u,v) ou false caso contrario.
    */
	public boolean existeAresta(int u, int v){
		if(this.adj[u][v] != 0){
			return true;
		}else{
			return false;
		}
	}

	/**
	 * Obtem o numero de vertices do digrafo.
    * @return o numero de vertices.
	 */
	public int numVertices(){
		return this.n;
	}

	/**
	 * Obtem o numero de arestas direcionadas do digrafo.
    * @return o numero de arestas do digrafo.
	 */
	public int numArestas(){
		int arestas = 0;
		for(int u = 0; u<numVertices(); u++){
			for(int v = 0; v<numVertices(); v++){
				if(existeAresta(u,v)){
					arestas++;
				}
			}
		}
		return arestas;
	}

	/**
	 * Verifica se existe algum loop no Digrafo.
    * @return true caso existir algum loop em qualquer vertice ou false caso contrario.
	 */
	public boolean existeLoop(){
		//FAZER
		return false; //resposta temporaria
	}

	/**
	 * Calcula o grau de saida do vertice u.
    * O grau de saida de um vertice u representa o numero de arestas direcionadas que possuem u como origem.
    * @return grau de saida do vertice u.
	 */
	public int grauSaida(int u){
		//FAZER
		return 0; //resposta temporaria
	}

	/**
	 * Calcula o grau de entrada do vertice u.
    * O grau de entrada de um vertice u representa o numero de arestas direcionadas que possuem u como destino.
    * @return grau de entrada do vertice u.
	 */
	public int grauEntrada(int u){
		//FAZER
		return 0; //resposta temporaria
	}

	/**
	 * Verifica se existe algum caminho com origem no vertice u e destino em v.
    * Os vertices u e v nao sao necessariamente adjacentes.
    * @return true caso existir algum de u para v ou false caso contrario.
	 */
	public boolean existeCaminho(int u, int v){
		//FAZER
		return false; //resposta temporaria
	}


}
