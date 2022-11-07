defmodule WebappWeb.PageControllerTest do
  use WebappWeb.ConnCase

  test "GET /", %{conn: conn} do
    conn = get(conn, ~p"/")
    assert html_response(conn, 200) =~ "Welcome to Phoenix!"
  end
end
