/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos;
 *         Lista de adjacencias.
 */

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * Classe que implementa um grafo direcionado, ou digrafo.
 */
public class ListaDigrafo extends Digrafo {
	private int n;
	private List<Integer> []adj;

   /**
    * Construtor de um digrafo de n vertices, sem nenhuma aresta.
    * As arestas devem ser adicionadas posteriormente.
	 * Os vertices sao rotulados de 0 ate n-1.
    */
	public ListaDigrafo(int n){
		this.n = n;
		this.adj = (List<Integer>[])new List[n];
		for(int u = 0; u<this.n; u++){
			this.adj[u] = new ArrayList<Integer>();
		}
	}

   /**
    * Adiciona uma nova aresta direcionada do vertice u para v.
    */
	public void adicionaAresta(int u, int v){
		this.adj[u].add(new Integer(v));
	}

   /**
    * Remove uma ocorrencia da aresta direcionada do vertice u para v, caso a mesma exista.
    */
	public void removeAresta(int u, int v){
		this.adj[u].remove(new Integer(v)); //remove uma ocorrencia da aresta de u para v.
	}

   /**
    * Verifica se existe uma aresta direcionada do vertice u para v.
    * @return true caso existir a aresta (u,v) ou false caso contrario.
    */
	public boolean existeAresta(int u, int v){
		if( this.adj[u].contains(new Integer(v)) ){
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
			arestas += adj[u].size();
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


	public void removerArestasParalelas(){
		for(int u = 0; u<numVertices(); u++){
			adj[u] = new ArrayList<Integer>(new HashSet<Integer>(adj[u]));
		}
	}
}
