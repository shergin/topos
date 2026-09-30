{
    use topos::Backend;

    let mut lines = Vec::new();
    for backend in Backend::ALL {
        match backend.status() {
            Ok(()) => lines.push(format!("{backend:?}: ready")),
            Err(reason) => lines.push(format!("{backend:?}: {reason}")),
        }
    }
    show::text(lines.join("\n"))
}
